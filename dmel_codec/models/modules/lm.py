from dataclasses import dataclass
from time import time
from typing import Optional, Tuple
from dmel_codec.models.modules.config_lm import Qwen2Config, FastQwen2ModelArgs
import torch
from torch import nn
from functools import partial
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RMSNorm
from transformers.modeling_outputs import ModelOutput
from transformers.loss.loss_utils import ForCausalLMLoss
from dmel_codec.utils.logger import RankedLogger
from einops import rearrange

log = RankedLogger(__name__, rank_zero_only=True)
SOFTMAX_IGNORE_INDEX = -100


@dataclass
class MultiModalCausalLMOutputWithPast(ModelOutput):
    text_logits: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    slow_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    fast_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    slow_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    fast_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    slow_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    fast_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    new_audio_labels: Optional[torch.LongTensor] = None


class ChatMusicSlowLMModel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.slow_lm_config = config
        self.pad_token_id = config.text_modality_mambaout_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.text_modality_mambaout_token_id,
        )

        self.slow_lm_audio_emb = nn.Embedding(
            config.audio_codebook_count * config.audio_codebook_size,
            config.hidden_size,
            config.slow_audio_modality_mambaout_token_id,
        )
        self.slow_audio_hiddenstate_projector = nn.Linear(
            config.hidden_size * config.audio_codebook_count,
            config.hidden_size,
            bias=False,
        )

        # special tokens
        self.start_of_human_id = config.start_of_human_id
        self.end_of_human_id = config.end_of_human_id
        self.start_of_robot_id = config.start_of_robot_id
        self.end_of_robot_id = config.end_of_robot_id
        self.start_of_music_id = config.start_of_music_id
        self.end_of_music_id = config.end_of_music_id

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ):
        return super().forward(
            inputs_embeds=inputs_embeds,
            *args,
            **kwargs,
        )

    def forward_generate(self, input_embeds, use_cache=True, past_key_values=None):
        """
            input_embeds: [bs, seq_len, hidden_size]
        """
        return super().forward(
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )


class ChatMusicFastLMModel(Qwen2Model):
    def __init__(self, config: FastQwen2ModelArgs):
        super().__init__(config)
        self.fast_lm_config = config
        self.need_project = config.slow_lm_hidden_size != config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.audio_pad_token_id, # pad means the unsame length audio pad
        )
        if self.need_project:
            self.slow_lm_to_fast_lm_dim_projector = nn.Linear(
                config.slow_lm_hidden_size, config.hidden_size
            )
        else:
            self.slow_lm_to_fast_lm_dim_projector = nn.Identity()

        self.pre_norm = Qwen2RMSNorm(
            config.slow_lm_hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, inp, labels):
        hidden_states = inp.last_hidden_state  # [bs, seq_len, slow_lm_hidden_size]

        # preprocess labels
        with torch.no_grad():  # 处理input，不涉及梯度计算

            # 去掉labels的T维度的第一个，对齐slow_lm的输出
            labels = labels[:, 1:, :]  # [bs, seq_len - 1, codebook_num]

            # 把-100替换为pad_token_id
            audio_inputs_ids = torch.where(
                labels == SOFTMAX_IGNORE_INDEX,
                self.fast_lm_config.audio_pad_token_id,
                labels,
            )

        # 去掉slow_lm_output的T维度的最后一个，对齐labels
        hidden_states = hidden_states[
            :, :-1, :
        ]  # [bs, seq_len - 1, slow_lm_hidden_size]
        # same with fishspeech
        hidden_states = self.pre_norm(hidden_states)

        if self.need_project:
            hidden_states = self.slow_lm_to_fast_lm_dim_projector(
                hidden_states
            )  # [bs, seq_len - 1, hidden_size]

        codebook_embeddings = self.embed_tokens(
            audio_inputs_ids
        )  # [bs, seq_len - 1, codebook_num, hidden_size]

        input_embeds = torch.cat(
            [hidden_states[:, :, None, :], codebook_embeddings], dim=2
        )  # [bs, seq_len - 1, codebook_num + 1, hidden_size]

        input_embeds = rearrange(input_embeds, "b s c h -> (b s) c h")
        outputs = super().forward(inputs_embeds=input_embeds)

        return outputs, labels

    def forward_generate(self, slow_hidden_state = None, fast_lm_ids = None, use_cache=False, fast_past_key_values=None):
        """
            slow_hidden_state: [1, seq_len(1: T+1), hidden_size]
            fast_lm_ids: None or [1 * seq_len(1: T+1), now_inference_codebook_num]
        """
        fast_lm_embeds = None
        if fast_lm_ids is not None:
            fast_lm_embeds = self.embed_tokens(fast_lm_ids)

        hidden_states = self.pre_norm(slow_hidden_state)
        if self.need_project:
            hidden_states = self.slow_lm_to_fast_lm_dim_projector(
                hidden_states
            ).unsqueeze(2)  # [bs, seq_len, 1, fast_lm_hidden_size]
            
        hidden_states = rearrange(hidden_states, "b s c h -> (b s) c h")
        if fast_lm_embeds is not None:
            hidden_states = torch.cat([hidden_states, fast_lm_embeds], dim=1) # [1 * T, now_inference_codebook_num + 1, hidden_size]

        fast_outputs = super().forward(
            inputs_embeds=hidden_states,
            use_cache=use_cache,
            past_key_values=fast_past_key_values,
        )

        return fast_outputs

class ChatMusicForCausalLM(nn.Module):
    def __init__(
        self,
        slow_lm_config: Qwen2Config,
        fast_lm_config: FastQwen2ModelArgs,
        text_weight: int = 1,
        audio_weight: int = 1,
    ):
        super().__init__()
        # new layers
        self.slow_model = ChatMusicSlowLMModel(slow_lm_config)
        self.fast_model = ChatMusicFastLMModel(fast_lm_config)

        self.text_lm_head = nn.Linear(
            slow_lm_config.hidden_size,
            slow_lm_config.vocab_size,
            bias=False,
        )

        # audio lm head
        self.audio_lm_head = nn.Linear(
            fast_lm_config.hidden_size,
            fast_lm_config.vocab_size,
            bias=False,
        )

        self.loss_function = partial(ForCausalLMLoss, ignore_index=-100)
        self.text_weight = text_weight
        self.audio_weight = audio_weight

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):
        text_outputs = self.slow_model(inputs_embeds=inputs_embeds)
        audio_outputs, audio_labels = self.fast_model(text_outputs, audio_labels)

        text_hidden_states = text_outputs["last_hidden_state"]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        text_logits = self.text_lm_head(text_hidden_states)

        audio_hidden_states = audio_outputs["last_hidden_state"]

        audio_logits = self.audio_lm_head(audio_hidden_states)  # [bs * (seq_len - 1), codebook_num + 1, vocab_size]

        text_loss = None
        audio_loss = None
        text_loss = self.loss_function(
            text_logits,
            text_labels,
            self.slow_model.vocab_size,
            **loss_kwargs,
        )

        if torch.isnan(text_loss) or torch.isinf(text_loss):
            text_loss -= text_loss
            log.info("Loss is nan or inf, setting to 0")

        # audio_logits: [bs * (seq_len - 1), codebook_num + 1, vocab_size]
        # audio_labels: [bs, (seq_len - 1), codebook_num], concat text shift labels to align timesteps
        tmp_text_labels = text_labels[:, 1:] # [bs, seq_len - 1]
        tmp_text_labels = tmp_text_labels.contiguous().view(-1, 1) # [bs * (seq_len - 1), 1]
        audio_labels = rearrange(audio_labels, "b s c -> (b s) c") # [bs * (seq_len - 1), codebook_num]
        audio_labels = torch.cat([tmp_text_labels, audio_labels], dim=1) # [bs * (seq_len - 1), codebook_num + 1]

        audio_loss = self.loss_function(
            audio_logits,
            audio_labels,
            self.fast_model.vocab_size,
            **loss_kwargs,
        )

        if torch.isnan(audio_loss) or torch.isinf(audio_loss):
            audio_loss -= audio_loss
            log.info("Audio Loss is nan or inf, setting to 0")

        weighted_loss = self.text_weight * text_loss + self.audio_weight * audio_loss

        return MultiModalCausalLMOutputWithPast(
            loss=weighted_loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            text_logits=text_logits,
            audio_logits=audio_logits,
            slow_past_key_values=text_outputs.past_key_values,
            fast_past_key_values=audio_outputs.past_key_values,
            slow_hidden_states=text_outputs.hidden_states,
            fast_hidden_states=audio_outputs.hidden_states,
            slow_attentions=text_outputs.attentions,
            fast_attentions=audio_outputs.attentions,
            new_audio_labels=audio_labels,
        )

    def forward_generate_text(self, input_embeds, use_cache=True, slow_past_key_values=None):
        """
            input_embeds: [1, seq_len, hidden_size]
        """
        text_outputs = self.slow_model.forward_generate(
            input_embeds=input_embeds,
            use_cache=use_cache,
            past_key_values=slow_past_key_values,
        )

        text_logits = self.text_lm_head(text_outputs.last_hidden_state)

        return MultiModalCausalLMOutputWithPast(
            loss=None,
            text_loss=None,
            audio_loss=None,
            text_logits=text_logits,
            audio_logits=None,
            slow_past_key_values=text_outputs.past_key_values,
            fast_past_key_values=None,
            slow_hidden_states=text_outputs.last_hidden_state,
            fast_hidden_states=None,
            slow_attentions=text_outputs.attentions,
            fast_attentions=None,
            new_audio_labels=None,
        )
        
    def forward_generate_audio(self, slow_hidden_state, fast_lm_ids = None, use_cache=False, fast_past_key_values=None):
        """
            slow_hidden_state: [1, seq_len(1: T+1), hidden_size]
            fast_lm_embeds: None or [1 * seq_len(1: T+1), now_inference_codebook_num, hidden_size]
        """
        audio_output = self.fast_model.forward_generate(
            slow_hidden_state=slow_hidden_state,
            fast_lm_ids=fast_lm_ids,
            use_cache=use_cache,
            fast_past_key_values=fast_past_key_values,
        )
        
        audio_logits = self.audio_lm_head(audio_output.last_hidden_state) # [1 * T, now_inference_codebook_num + 1, vocab_size]

        return MultiModalCausalLMOutputWithPast(
            loss=None,
            text_loss=None,
            audio_loss=None,
            text_logits=None,
            audio_logits=audio_logits,
            slow_past_key_values=None,
            fast_past_key_values=audio_output.past_key_values,
            slow_hidden_states=None,
            fast_hidden_states=audio_output.hidden_states,
            slow_attentions=None,
            fast_attentions=audio_output.attentions,
            new_audio_labels=None,
        )

if __name__ == "__main__":
    device = "cpu"
    slow_lm_config = Qwen2Config.from_pretrained(
        "/home/wzy/projects/dmel_codec/dmel_codec/config/lm/slow_lm_0.5B.json"
    )
    fast_lm_config = FastQwen2ModelArgs.from_pretrained(
        "/home/wzy/projects/dmel_codec/dmel_codec/config/lm/fast_lm.json"
    )
    model = ChatMusicForCausalLM(slow_lm_config, fast_lm_config)
    model.to(device)
    fake_text_inputs_ids = torch.randint(0, 151650, (3, 20)).to(device)

    # codebook 10, codebook shift 175
    fake_audio_list = []
    for i in range(10):
        fake_audio_list.append(torch.randint(175 * i, 175 * (i + 1), (3, 20)))
    fake_audio_inputs_ids = torch.stack(fake_audio_list, dim=2).to(device)
    with torch.no_grad():
        print(
            model(
                text_inputs_ids=fake_text_inputs_ids,
                audio_inputs_ids=fake_audio_inputs_ids,
            )
        )