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
from utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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
        self.slow_lm_audio_emb = nn.Embedding(config.audio_codebook_count * config.audio_codebook_size, config.hidden_size)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class ChatMusicFastLMModel(Qwen2Model):
    def __init__(self, config: FastQwen2ModelArgs):
        super().__init__(config)
        self.fast_lm_config = config
        self.need_project = (config.slow_lm_hidden_size != config.hidden_size)
        if self.need_project:
            self.slow_lm_to_fast_lm_dim_projector = nn.Linear(config.slow_lm_hidden_size, config.hidden_size)
        else:
            self.slow_lm_to_fast_lm_dim_projector = nn.Identity()

    def forward(self, inp, labels):
        hidden_states = inp.last_hidden_state
        
        if self.need_project:
            hidden_states = self.slow_lm_to_fast_lm_dim_projector(hidden_states) # [bs, seq_len, hidden_size]
        
        codebook_embeddings = self.embed_tokens(labels.clone()) # [bs, seq_len, codebook_num, hidden_size]
        input_embeds = torch.cat([hidden_states[:, :, None, :], codebook_embeddings], dim=2) # [bs, seq_len, codebook_num + 1, hidden_size]
        b, s, c, h = input_embeds.shape
        input_embeds = input_embeds.view(b * s, c, h)
        outputs = super().forward(inputs_embeds = input_embeds)
        
        return outputs, labels
        


class ChatMusicForCausalLM(nn.Module):
    def __init__(self, slow_lm_config: Qwen2Config, fast_lm_config: FastQwen2ModelArgs):

        # new layers
        self.slow_model = ChatMusicSlowLMModel(slow_lm_config)
        self.fast_model = ChatMusicFastLMModel(fast_lm_config)

        self.audio_lm_head = nn.Linear(
            fast_lm_config.hidden_size,
            fast_lm_config.codebook_nums * fast_lm_config.audio_codebook_size,
        )
        self.loss_function = partial(ForCausalLMLoss, ignore_index=-100)


    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):

        text_outputs = self.slow_model(inputs_embeds=inputs_embeds)
        audio_outputs = self.fast_model(text_outputs, audio_labels)

        text_hidden_states = text_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        text_logits = self.lm_head(text_hidden_states)
        
        audio_hidden_states = audio_outputs[0]
        audio_logits = audio_hidden_states

        audio_logits = audio_logits.view(
            audio_logits.shape[0],
            audio_logits.shape[1],
            -1,
            self.config.audio_codebook_size,
        ).transpose(1, 2)

        text_loss = None
        audio_loss = None

        if text_labels is not None:
            text_loss = self.loss_function(
                text_logits, text_labels, self.vocab_size, **loss_kwargs
            )

            if torch.isnan(text_loss) or torch.isinf(text_loss):
                text_loss -= text_loss
                log.info("Loss is nan or inf, setting to 0")

        if audio_labels is not None:
            audio_loss = self.loss_function(
                audio_logits,
                audio_labels.transpose(1, 2),
                self.config.audio_codebook_size,
                **loss_kwargs,
            )
            if torch.isnan(audio_loss) or torch.isinf(audio_loss):
                audio_loss -= audio_loss
                log.info("Audio Loss is nan or inf, setting to 0")

        weighted_loss = None
        if text_loss is not None and audio_loss is not None:
            weighted_loss = (
                self.config.text_loss_weight * text_loss
                + self.config.audio_loss_weight * audio_loss
            )

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
    slow_lm_config = Qwen2Config.from_pretrained("/sdb/model_weight/qwen2-0.5B")
    fast_lm_config = FastQwen2ModelArgs.from_pretrained("/home/wzy/projects/dmel_codec/dmel_codec/config/lm/fast_lm.json")
    model = ChatMusicForCausalLM(slow_lm_config, fast_lm_config)
    print(model)