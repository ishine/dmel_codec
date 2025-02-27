import lightning.pytorch as pl
from typing import Any, Optional
from torch import nn as nn
from dmel_codec.models.modules.config_lm import Qwen2Config, FastQwen2ModelArgs
from dmel_codec.models.modules.lm import (
    ChatMusicForCausalLM,
    MultiModalCausalLMOutputWithPast,
)
from dmel_codec.utils.logger import RankedLogger
from dmel_codec.models.modules.lm_process_input import ProcessInputs
from safetensors.torch import load_file
import torch
import os
from transformers.models.qwen2 import Qwen2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from time import time
from einops import rearrange
from dmel_codec.utils.utils import sample_one_token_from_logits

SOFTMAX_IGNORE_INDEX = -100
log = RankedLogger(__name__, rank_zero_only=True)


class MusicLLM(pl.LightningModule):
    def __init__(
        self,
        slow_lm_config_path: str,
        fast_lm_config_path: str,
        # Codec
        codec_model: nn.Module,
        # Other parameters
        silence_length: int,
        audio_silence_id: list,
        max_length: int,
        optimizer: Any = None,
        lr_scheduler: Any = None,
        is_strict: bool = False,
        text_foundation_model_path: str | None = None,
        text_tokenizer_path: str | None = None,
        mllm_model_path: str | None = None,
        model_dtype: str = "bfloat16",
        codec_ckpt_path: str | None = None,
        text_weight: int = 1,
        audio_weight: int = 1,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
    ):
        super().__init__()

        slow_lm_config = Qwen2Config.from_pretrained(slow_lm_config_path)
        fast_lm_config = FastQwen2ModelArgs.from_pretrained(fast_lm_config_path)
        self.automatic_optimization = False
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.strict_loading = is_strict
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        model = ChatMusicForCausalLM(
            slow_lm_config=slow_lm_config,
            fast_lm_config=fast_lm_config,
            text_weight=text_weight,
            audio_weight=audio_weight,
        )
        self.model = model
        if mllm_model_path is None:
            assert text_foundation_model_path is not None
            log.info("Loading qwen2 text base model from pretrained checkpoints")
            if text_foundation_model_path.endswith(".safetensors"):
                self.model.load_state_dict(
                    load_file(text_foundation_model_path, device="cpu"), strict=False
                )

            else:
                qwen_pretrain_state_dict = load_file(
                    os.path.join(text_foundation_model_path, "model.safetensors"),
                    device="cpu",
                )
                qwen_pretrain_state_dict = self.switch_qwen_hf_2_ours(qwen_pretrain_state_dict)
                self.model.load_state_dict(
                    qwen_pretrain_state_dict,
                    strict=False,
                )

                # ensure the padding_idx is 0
                self.model.slow_model.embed_tokens._fill_padding_idx_with_zero()
            log.info(
                f"Loading qwen2 mllm model from {text_foundation_model_path} successfully"
            )

        else:
            log.info("Loading qwen2 mllm model from pretrained checkpoints")
            self.load_state_dict(
                torch.load(mllm_model_path, map_location="cpu")["state_dict"],
                strict=False,
            )
            log.info(f"Loading qwen2 mllm model from {mllm_model_path} successfully")

        self.max_length = max_length
        self.silence_length = silence_length
        text_tokenizer = Qwen2Tokenizer.from_pretrained(text_tokenizer_path)

        # codec
        codec_model.load_state_dict(
            torch.load(codec_ckpt_path, map_location="cpu")["state_dict"], strict=False
        )
        self.codec_model = codec_model
        log.info(f"Loading codec model from {codec_ckpt_path} successfully")
        for _, param in self.codec_model.named_parameters():
            param.requires_grad = False

        if model_dtype == "bfloat16":
            self.codec_model.to(torch.bfloat16)
            self.model.to(torch.bfloat16)
        elif model_dtype == "float16":
            self.codec_model.to(torch.float16)
            self.model.to(torch.float16)
        else:
            raise ValueError(f"Unsupported model dtype: {model_dtype}, please use 'bfloat16' or 'float16'")

        self.slow_lm_config: Qwen2Config = slow_lm_config
        self.fast_lm_config: FastQwen2ModelArgs = fast_lm_config

        self.process_inputs_cls = ProcessInputs(
            config=slow_lm_config,
            max_length=max_length,
            silence_length=silence_length,
            audio_silence_id=audio_silence_id,
            text_tokenizer=text_tokenizer,
        )
    
    def switch_qwen_hf_2_ours(self, state_dict):
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("model.", "slow_model.")
            new_state_dict[new_key] = state_dict[key]
        return new_state_dict

    def get_accuracy(
        self,
        logits,
        labels,
        ignore_index=[SOFTMAX_IGNORE_INDEX, 179],
        topk: list[int] = [1, 5],
    ):
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        accuracy_list = []

        # 创建掩码，标记不需要忽略的位置
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
        for ignore_id in ignore_index:
            valid_mask &= labels != ignore_id

        for k in topk:
            _, indices = logits.topk(k, dim=-1)
            correct = indices.eq(labels.unsqueeze(-1))
            for ignore_id in ignore_index:
                correct[labels == ignore_id] = 0
            correct = correct.sum()
            # 使用valid_mask计算有效标签的数量
            accuracy = correct / valid_mask.sum()
            accuracy_list.append(accuracy)
        return accuracy_list

    # def on_after_backward(self):
    #     # 打印所有参数的梯度
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"Gradient for {name}: {param.grad}")

    def configure_optimizers(self):
        # Get weight decay parameters
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        for i in optimizer.param_groups:
            log.info(
                f"Set weight decay: {i['weight_decay']} for {len(i['params'])} parameters"
            )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def get_embeds_from_inputs_ids(self, text_ids, audio_ids):
        """
            text_ids: [T]
            audio_ids: [T, codebook_num]
        """
        text_embeds = self.model.slow_model.embed_tokens(text_ids) # shape = (T, D)

        audio_inputs_embeds = self.model.slow_model.slow_lm_audio_emb(audio_ids) # shape = (T, codebook_num, D)

        audio_inputs_embeds = rearrange(audio_inputs_embeds, "s c h -> s (c h)") # shape = (T, codebook_num * D)

        audio_embeds = self.model.slow_model.slow_audio_hiddenstate_projector(
            audio_inputs_embeds
        )

        return text_embeds + audio_embeds # shape = (T, D)

    def process_all_input_for_train(self, batch):
        inputs_embeds_list = []
        labels_list = []
        audio_ids_list = self.process_inputs_cls.get_audio_ids_parralel(
            batch["audios"], batch["audio_lengths"], self.codec_model
        )

        for i in range(len(batch["text"])):
            item = {
                "text": batch["text"][i],
                "audio_ids": audio_ids_list[i],
            }
            (text_modality_tokens, audio_modality_tokens, labels) = (
                self.process_inputs_cls.get_input_label(item, self.codec_model.device)
            )

            labels_list.append(labels)
            input_embeds = self.get_embeds_from_inputs_ids(
                text_modality_tokens, audio_modality_tokens
            )
            inputs_embeds_list.append(input_embeds)

        inputs_embeds = pad_sequence(
            inputs_embeds_list, batch_first=True, padding_value=0
        ) # shape = (bs, T, D)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=SOFTMAX_IGNORE_INDEX
        ) # shape = (bs, T, codebook_num + 1)
        return inputs_embeds, labels[:, :, 0], labels[:, :, 1:]

    def _step(self, batch, batch_idx, stage="train"):
        is_train = stage == "train"
        if is_train:
            self.model.train()
            optimizer = self.optimizers()
            lr_scheduler = self.lr_schedulers()
        else:
            self.model.eval()

        inputs_embeds, text_labels, audio_labels = self.process_all_input_for_train(
            batch
        )

        multimodel_causual_output: MultiModalCausalLMOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            text_labels=text_labels,
            audio_labels=audio_labels,
        )

        self.log(
            f"{stage}/llm_loss",
            multimodel_causual_output.loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
            rank_zero_only=True,
        )

        self.log(
            f"{stage}/text_loss",
            multimodel_causual_output.text_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
            rank_zero_only=True,
        )

        self.log(
            f"{stage}/audio_loss",
            multimodel_causual_output.audio_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
            rank_zero_only=True,
        )

        topk_list = [1, 2, 5, 10, 20, 50]
        # 忽略mambaout_token_id 和 ignore_index
        accuracy_list = self.get_accuracy(
            multimodel_causual_output.audio_logits,
            multimodel_causual_output.new_audio_labels,
            ignore_index=[
                SOFTMAX_IGNORE_INDEX,
                self.slow_lm_config.slow_audio_modality_mambaout_token_id,
            ],
            topk=topk_list,
        )
        for k, accuracy in zip(topk_list, accuracy_list):
            self.log(
                f"{stage}/top_{k}_accuracy",
                accuracy,
                on_step=is_train,
                on_epoch=not is_train,
                prog_bar=True,
                logger=True,
                sync_dist=not is_train,
                rank_zero_only=True,
            )
        loss = multimodel_causual_output.loss

        if is_train:
            # 梯度累积处理
            accumulated_loss = loss / self.accumulate_grad_batches
            accumulated_loss.backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val is not None:
                    if self.gradient_clip_algorithm == "norm":
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=self.gradient_clip_val,
                            error_if_nonfinite=True
                        )
                    elif self.gradient_clip_algorithm == "value":
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(), 
                            clip_value=self.gradient_clip_val
                        )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            return {"loss": loss.detach()}
        else:
            return loss

    def on_train_batch_end(self, outputs: torch.Tensor | os.Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if (batch_idx+1) % (self.accumulate_grad_batches * 3) == 0: # per 3 steps empty cache
            torch.cuda.empty_cache()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        try:
            return self._step(batch, batch_idx, stage="train")
        except Exception as _:
            return {"loss": torch.tensor(0.0).requires_grad_(True).detach()}

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self._step(batch, batch_idx, stage="val")

    def on_save_checkpoint(self, checkpoint):
        if self.slow_lm_config.lora_config is not None:
            # Save only LoRA parameters
            state_dict = checkpoint["state_dict"]
            for name in list(state_dict.keys()):
                if "lora" not in name:
                    state_dict.pop(name)
        else:
            for name in list(checkpoint["state_dict"].keys()):
                if "codec_model" in name:
                    checkpoint["state_dict"].pop(name)

    # -------------------------- Inference --------------------------
    @torch.no_grad()
    @torch.inference_mode()
    def inference_by_audio_prompt(self, inference_config, wav):
        audio_logits, _ = self.codec_model.encode(
            wav.unsqueeze(0),
            torch.tensor(wav.shape[-1]).reshape(
                -1,
            ),
        )
        audio_logits = audio_logits.squeeze(0)
        self.audio_prompt_length = audio_logits.shape[-1]
        self.audio_logits = audio_logits  # shape = (codebook_count, T)
        self.text_logits = None
        self.text_prompt_length = 0
        self.codebook_shift = (
            torch.arange(self.config.audio_codebook_count)
            * self.config.audio_codebook_size
        ).to(self.model.device)
        input_ids = self.process_2d_logits(
            self.text_logits, self.audio_logits, mode="infer"
        ).T.unsqueeze(0)

        next_prefilling_token, past_kv = self.prefilling_next_token(
            input_ids,
            past_key_values=None,
            previous_token=None,
            inference_config=inference_config,
        )
        input_ids_ar = self.process_input_ids_inference(
            next_prefilling_token.squeeze(1)
        )
        self.predict_n_token(input_ids_ar, past_kv, inference_config)

        wav, _ = self.codec_model.decode(
            self.audio_logits.unsqueeze(0),
            torch.tensor(self.audio_logits.shape[-1])
            .to(self.codec_model.device)
            .unsqueeze(0),
            return_audios=True,
        )

        return wav.float().cpu().numpy()

    @torch.inference_mode()
    def inference_by_text_prompt(self, inference_config):
        text_ids = self.text_inference_input_processor(inference_config) # shape = (1, T)
        self.text_prompt_length = text_ids.shape[1]
        self.audio_prompt_length = 0

        self.codebook_shift = (
            torch.arange(self.slow_lm_config.audio_codebook_count)
            * self.slow_lm_config.audio_codebook_size
        ).to(self.device)
        input_ids = self.process_inputs_cls.process_2d_logits_infer(
            device=self.device,
            text_ids=text_ids,
            audio_ids=None,
            text_prompt_length=self.text_prompt_length,
            audio_prompt_length=0,
        ).T  # (S, Codebook_num)
        text_input_ids, audio_input_ids = input_ids[:, 0], input_ids[:, 1:]
        self.text_ids = text_input_ids
        self.audio_ids = audio_input_ids
        input_embeds = self.get_embeds_from_inputs_ids(
            text_input_ids, audio_input_ids
        ).unsqueeze(0)  # (1, S, H)
        slow_past_key_values = None

        # prefilling
        next_prefilling_token, slow_past_key_values = (
            self.prefilling_next_token(
                input_embeds=input_embeds,
                slow_past_key_values=slow_past_key_values,
                previous_token=None,
                inference_config=inference_config,
            )
        )
        input_embeds_ar = self.process_generation_ids(next_prefilling_token)

        self.predict_n_token(input_embeds_ar, slow_past_key_values, inference_config)
        generation_audio_ids = self.audio_ids[self.text_prompt_length + 6:-1]
        # deshift audio ids
        generation_audio_ids = generation_audio_ids - self.codebook_shift
        wav, _ = self.codec_model.decode(
            indices=generation_audio_ids.T.unsqueeze(0), # (T, Codebook_num) -> (1, Codebook_num, T)
            feature_lengths=torch.tensor(generation_audio_ids.shape[0]).to(self.codec_model.device).unsqueeze(0),
            return_audios=True,
        )
        return wav.float().cpu().squeeze(0)

    def predict_one_token(
        self,
        input_embeds,
        slow_past_key_values,
        inference_config,
        previous_token=None,
    ):
        text_output_dict = self.sample_text_token(
            input_embeds=input_embeds,
            inference_config=inference_config,
            slow_past_key_values=slow_past_key_values,
        )
        slow_past_key_values = text_output_dict["slow_past_key_values"]
        slow_lm_hidden_state = text_output_dict["slow_hidden_state"]
        text_token = text_output_dict["text_token"]

        audio_token_generate_list = []
        # audio generation
        for i in range(self.fast_lm_config.codebook_nums):
            if i == 0:  # semantic to first codebook
                audio_token = self.sample_audio_token(
                    inference_config=inference_config,
                    previous_token=previous_token[:, i:i+1].squeeze(1) if previous_token is not None else None,
                    slow_lm_hidden_state=slow_lm_hidden_state,
                    fast_lm_ids=None,
                    use_cache=False,
                    fast_past_key_values=None, # don't use kv cache in the fast lm
                )
                audio_token_generate_list.append(audio_token)
            else:
                pad_ids_for_inference = self.audio_ids[1:, :i]  # (bs * seq_len - 1, i) # text_output is the next token, so we need to shift the audio ids for alignment
                audio_generated_ids = torch.tensor(audio_token_generate_list, device=pad_ids_for_inference.device, dtype=pad_ids_for_inference.dtype).unsqueeze(0)  # (1, i)
                audio_generated_ids = torch.cat([pad_ids_for_inference, audio_generated_ids], dim=0)  # (bs * seq_len, i)

                audio_token = self.sample_audio_token(
                    inference_config=inference_config,
                    previous_token=previous_token[:, i:i+1].squeeze(1) if previous_token is not None else None,
                    slow_lm_hidden_state=slow_lm_hidden_state,
                    fast_lm_ids=audio_generated_ids,
                    use_cache=False,
                    fast_past_key_values=None,
                )
                audio_token_generate_list.append(audio_token)

        return torch.stack([text_token] + audio_token_generate_list, dim=0), slow_past_key_values

    def prefilling_next_token(
        self,
        input_embeds,
        slow_past_key_values,
        previous_token,
        inference_config,
    ):
        return self.predict_one_token(
            input_embeds=input_embeds,
            slow_past_key_values=slow_past_key_values,
            previous_token=previous_token,
            inference_config=inference_config,
        )

    def predict_n_token(self, input_embeds, slow_past_key_values, inference_config):
        now_time_step, _ = self.audio_ids.shape

        previous_tokens = torch.zeros(
            (self.max_length, self.slow_lm_config.audio_codebook_count),
            dtype=torch.long,
            device=input_embeds.device) # shape = (T, Codebook_num)

        previous_tokens[:now_time_step, :] = self.audio_ids.clone()
        input_embeds_ar = input_embeds

        while self.is_end_of_predict(now_time_step, inference_config) is False:
            win_size = inference_config.windows_length

            if now_time_step < win_size:
                window = previous_tokens[:now_time_step, :]
            else:
                window = previous_tokens[now_time_step - win_size:, :]

            next_token, slow_past_key_values = self.predict_one_token(
                input_embeds=input_embeds_ar,
                slow_past_key_values=slow_past_key_values,
                inference_config=inference_config,
                previous_token=window,
            )

            now_time_step += 1
            previous_tokens[now_time_step, :] = next_token[1:, :].squeeze(1).clone() # next_token[0, :] is text token
            input_embeds_ar = self.process_generation_ids(next_token)

    def sample_text_token(
        self, input_embeds, inference_config, slow_past_key_values=None
    ):
        text_output: MultiModalCausalLMOutputWithPast = (
            self.model.forward_generate_text(
                input_embeds=input_embeds,
                use_cache=True,
                slow_past_key_values=slow_past_key_values,
            )
        )

        text_logits = text_output.text_logits

        text_token = sample_one_token_from_logits(
            logits=text_logits[0, -1, :], # only one token need to sample
            previous_token=None,  # text token ignore windows_penalty
            temperature=inference_config.temperature,
            top_k=inference_config.top_k,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.windows_penalty,
        )[0]
        return {
            "text_token": text_token,
            "slow_past_key_values": text_output.slow_past_key_values,
            "slow_hidden_state": text_output.slow_hidden_states,
        }

    def sample_audio_token(
        self,
        inference_config,
        previous_token=None,
        slow_lm_hidden_state=None,
        fast_lm_ids=None,
        use_cache=False,
        fast_past_key_values=None,
    ):
        assert (
            slow_lm_hidden_state is not None or fast_lm_ids is not None
        ), "slow_lm_hidden_state and fast_lm_ids cannot be both None"
        audio_output: MultiModalCausalLMOutputWithPast = (
            self.model.forward_generate_audio(
                slow_hidden_state=slow_lm_hidden_state,
                fast_lm_ids=fast_lm_ids,
                use_cache=use_cache,
                fast_past_key_values=fast_past_key_values,
            )
        )
        audio_logits = audio_output.audio_logits  # (bs * seq_len, 1, vocab_size)
        audio_token = sample_one_token_from_logits(
            logits=audio_logits[-1, -1, :], # only one token need to sample
            previous_token=previous_token,
            temperature=inference_config.temperature,
            top_k=inference_config.top_k,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.windows_penalty,
        )[0]
        return audio_token

    def is_end_of_predict(self, cur_generation_token_nums, inference_config):
        return (self.text_ids[-1].item() == self.slow_lm_config.end_of_music_id) or \
                (cur_generation_token_nums >= min(self.max_length, inference_config.max_new_tokens))

    def text_inference_input_processor(self, inference_config):
        text_logits = self.process_inputs_cls.text_tokenizer(
            inference_config.prompt, return_tensors="pt"
        )["input_ids"].to(self.device)

        # text_special_token_start = torch.tensor([self.config.start_of_human_id, self.config.bos_token_id],
        #                                              dtype = torch.long).to(self.model.device)

        # text_special_token_middle_list = torch.tensor([self.config.eos_token_id,
        #                                   self.config.end_of_human_id,
        #                                   self.config.start_of_robot_id,
        #                                   self.config.start_of_music_id], dtype=torch.long).to(self.model.device)

        # input_ids = torch.cat([text_special_token_start, text_logits, text_special_token_middle_list], dim=0)
        return text_logits

    def process_generation_ids(self, next_token):
        """
            next_token: [codebook_num + 1, 1]: codebook_num + 1 means text_token + codebook_num
        """
        self.text_ids = (
            torch.cat(
                [self.text_ids, next_token[0]], dim=0
            )
            if self.text_ids is not None
            else next_token[0]
        ) # shape = (T)

        self.audio_ids = (
            torch.cat(
                [self.audio_ids, next_token[1:].T], dim=0
            ).squeeze(0)
            if self.audio_ids is not None
            else next_token[1:].T
        ) # shape = (T, Codebook_num)
        
        input_embeds = self.get_embeds_from_inputs_ids(
            self.text_ids, self.audio_ids
        ).unsqueeze(0) # shape = (1, T, H)
        return input_embeds


if __name__ == "__main__":
    # test train code
    from dmel_codec.dataset.lhotse_tts_dataset import LhotseDataModule
    from functools import partial
    from dmel_codec.utils.schedule import get_cosine_schedule_with_warmup_lr_lambda
    import hydra
    from omegaconf import OmegaConf

    dataset = LhotseDataModule(
        stage="validate",
        val_cuts_path="/home/wzy/projects/dmel_codec/val_cuts_sample-128.jsonl.gz",
        world_size=1,
        val_max_durations=40,
        val_num_workers=1,
    )
    dataset.setup("validate")
    val_dataloader = dataset.val_dataloader()
    optimizer = partial(
        torch.optim.AdamW, lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01
    )
    cosine_warmup = partial(
        get_cosine_schedule_with_warmup_lr_lambda,
        num_training_steps=1000000,
        num_warmup_steps=2000,
        final_lr_scale=0.01,
    )
    lr_scheduler = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=cosine_warmup)

    config = OmegaConf.load(
        "/home/wzy/projects/dmel_codec/dmel_codec/config/lm/lm_config.yaml"
    )
    codec_model = hydra.utils.instantiate(config.model.codec_model, _convert_="partial")
    model = MusicLLM(
        slow_lm_config_path="/home/wzy/projects/dmel_codec/dmel_codec/config/lm/slow_lm_0.5B.json",
        fast_lm_config_path="/home/wzy/projects/dmel_codec/dmel_codec/config/lm/fast_lm.json",
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        codec_model=codec_model,
        silence_length=3,
        audio_silence_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        max_length=4096,
        is_strict=False,
        text_foundation_model_path="/sdb/model_weight/qwen2-0.5B",
        text_tokenizer_path="/sdb/model_weight/qwen2-0.5B",
        mllm_model_path=None,
    )
    for batch in val_dataloader:
        a = model._step(batch, 0)
        print(a)
