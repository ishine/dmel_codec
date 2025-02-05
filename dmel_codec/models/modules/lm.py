from dataclasses import dataclass
from time import time
from typing import Optional, Tuple
from dmel_codec.models.modules.config_lm import Qwen2Config, FastQwen2ModelArgs
import torch
from torch import nn
from functools import partial
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.modeling_outputs import ModelOutput
from transformers.loss.loss_utils import ForCausalLMLoss
from dmel_codec.utils.logger import RankedLogger

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
        # text_inputs_ids: Optional[torch.LongTensor],
        # audio_inputs_ids: Optional[torch.LongTensor],
    ):
        # text_inputs_embeds = self.embed_tokens(text_inputs_ids)
        # audio_inputs_embeds = self.slow_lm_audio_emb(audio_inputs_ids)

        # bs, seq_len, codebook_num, hidden_size = audio_inputs_embeds.shape
        # audio_inputs_embeds = audio_inputs_embeds.view(
        #     bs, seq_len, codebook_num * hidden_size
        # )
        # audio_inputs_embeds = self.slow_audio_hiddenstate_projector(audio_inputs_embeds)

        # inputs_embeds = text_inputs_embeds + audio_inputs_embeds
        return super().forward(inputs_embeds=inputs_embeds)


class ChatMusicFastLMModel(Qwen2Model):
    def __init__(self, config: FastQwen2ModelArgs):
        super().__init__(config)
        self.fast_lm_config = config
        self.need_project = config.slow_lm_hidden_size != config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.audio_pad_token_id,
        )
        if self.need_project:
            self.slow_lm_to_fast_lm_dim_projector = nn.Linear(
                config.slow_lm_hidden_size, config.hidden_size
            )
        else:
            self.slow_lm_to_fast_lm_dim_projector = nn.Identity()

    def forward(self, inp, labels):
        hidden_states = inp.last_hidden_state # [bs, seq_len, slow_lm_hidden_size]
        
        with torch.no_grad(): # 处理input，不涉及梯度计算
            audio_inputs_ids = torch.where(
                labels == SOFTMAX_IGNORE_INDEX,
                self.fast_lm_config.audio_pad_token_id,
                labels,
            )

        if self.need_project:
            hidden_states = self.slow_lm_to_fast_lm_dim_projector(
                hidden_states
            )  # [bs, seq_len, hidden_size]

        codebook_embeddings = self.embed_tokens(
            audio_inputs_ids
        )  # [bs, seq_len, codebook_num, hidden_size]
        input_embeds = torch.cat(
            [hidden_states[:, :, None, :], codebook_embeddings], dim=2
        )  # [bs, seq_len, codebook_num + 1, hidden_size]
        b, s, c, h = input_embeds.shape
        input_embeds = input_embeds.view(b * s, c, h)
        outputs = super().forward(inputs_embeds=input_embeds)

        outputs["last_hidden_state"] = outputs["last_hidden_state"].view(b, s, c, h)

        return outputs


class ChatMusicForCausalLM(nn.Module):
    def __init__(self, slow_lm_config: Qwen2Config, fast_lm_config: FastQwen2ModelArgs):
        super().__init__()
        # new layers
        self.slow_model = ChatMusicSlowLMModel(slow_lm_config)
        self.fast_model = ChatMusicFastLMModel(fast_lm_config)

        self.text_lm_head = nn.Linear(
            slow_lm_config.hidden_size,
            slow_lm_config.vocab_size,
            bias=False,
        )

        self.audio_lm_head = nn.Linear(
            fast_lm_config.hidden_size,
            fast_lm_config.vocab_size,
            bias=False,
        )
        self.loss_function = partial(ForCausalLMLoss, ignore_index=-100)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # text_inputs_ids: Optional[torch.LongTensor] = None,
        # audio_inputs_ids: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):
        # text_inputs_ids: [bs, seq_len]
        # audio_inputs_ids: [bs, seq_len, codebook_num]

        # text_outputs = self.slow_model(
        #     text_inputs_ids=text_inputs_ids, audio_inputs_ids=audio_inputs_ids
        # )
        text_outputs = self.slow_model(inputs_embeds=inputs_embeds)
        audio_outputs = self.fast_model(text_outputs, audio_labels)

        text_hidden_states = text_outputs["last_hidden_state"]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        text_logits = self.text_lm_head(text_hidden_states)

        audio_hidden_states = audio_outputs["last_hidden_state"]
        audio_logits = self.audio_lm_head(audio_hidden_states)

        text_loss = None
        audio_loss = None

        text_loss = self.loss_function(
            text_logits,
            text_labels,
            text_logits.shape[-1],
            **loss_kwargs,
        )

        if torch.isnan(text_loss) or torch.isinf(text_loss):
            text_loss -= text_loss
            log.info("Loss is nan or inf, setting to 0")

        # audio_logits: [bs, seq_len, codebook_num + 1, vocab_size], 1 means the text modality semantic token
        # audio_labels: [bs, seq_len, codebook_num], so we need to expand the first dim of audio_labels to [bs, seq_len, codebook_num + 1]
        audio_labels = torch.cat([text_labels[:, :, None], audio_labels], dim=2)
        audio_loss = self.loss_function(
            audio_logits,
            audio_labels,
            audio_logits.shape[-1],
            **loss_kwargs,
        )
        if torch.isnan(audio_loss) or torch.isinf(audio_loss):
            audio_loss -= audio_loss
            log.info("Audio Loss is nan or inf, setting to 0")

        # TODO: add weighted loss
        # weighted_loss = None
        # if text_loss is not None and audio_loss is not None:
        #     weighted_loss = (
        #         self.slow_lm_config.text_loss_weight * text_loss
        #         + self.fast_lm_config.audio_loss_weight * audio_loss
        #     )
        weighted_loss = text_loss + audio_loss

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
