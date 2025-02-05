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
    ):
        super().__init__()

        slow_lm_config = Qwen2Config.from_pretrained(slow_lm_config_path)
        fast_lm_config = FastQwen2ModelArgs.from_pretrained(fast_lm_config_path)

        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.strict_loading = is_strict
        model = ChatMusicForCausalLM(slow_lm_config, fast_lm_config)
        if mllm_model_path is None:
            assert text_foundation_model_path is not None
            log.info("Loading qwen2 text base model from pretrained checkpoints")
            if text_foundation_model_path.endswith(".safetensors"):
                model.load_state_dict(
                    load_file(text_foundation_model_path, device="cpu"), strict=False
                )

            else:
                model.load_state_dict(
                    load_file(
                        os.path.join(text_foundation_model_path, "model.safetensors"),
                        device="cpu",
                    ),
                    strict=False,
                )
            log.info(
                f"Loading qwen2 mllm model from {text_foundation_model_path} successfully"
            )

        else:
            log.info("Loading qwen2 mllm model from pretrained checkpoints")

        self.model = model

        self.max_length = max_length
        self.silence_length = silence_length
        text_tokenizer = Qwen2Tokenizer.from_pretrained(text_tokenizer_path)

        # codec
        self.codec_model = codec_model
        for _, param in self.codec_model.named_parameters():
            param.requires_grad = False
            param.to()

        if model_dtype == "bfloat16":
            self.codec_model.to(torch.bfloat16)
            self.model.to(torch.bfloat16)
        elif model_dtype == "float16":
            self.codec_model.to(torch.float16)
            self.model.to(torch.float16)
        else:
            raise ValueError(f"Unsupported model dtype: {model_dtype}")

        self.slow_lm_config: Qwen2Config = slow_lm_config
        self.fast_lm_config: FastQwen2ModelArgs = fast_lm_config

        self.process_inputs_cls = ProcessInputs(
            config=slow_lm_config,
            max_length=max_length,
            silence_length=silence_length,
            audio_silence_id=audio_silence_id,
            text_tokenizer=text_tokenizer,
        )

    def get_accuracy(self, logits, labels, ignore_index=[SOFTMAX_IGNORE_INDEX, 179], topk: list[int]=[1, 5]):
        logits = logits[:, :, :-1, :] # token shift
        accuracy_list = []
        
        # 创建掩码，标记不需要忽略的位置
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
        for ignore_id in ignore_index:
            valid_mask &= (labels != ignore_id)

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

    def get_embeds_from_inputs_ids(self, text_logits, audio_logits):
        """
        text_logits: [T]
        audio_logits: [T, codebook_num]
        """
        text_embeds = self.model.slow_model.embed_tokens(text_logits)

        audio_inputs_embeds = self.model.slow_model.slow_lm_audio_emb(audio_logits)
        seq_len, codebook_num, hidden_size = audio_inputs_embeds.shape
        audio_inputs_embeds = audio_inputs_embeds.view(
            seq_len, codebook_num * hidden_size
        )

        audio_embeds = self.model.slow_model.slow_audio_hiddenstate_projector(
            audio_inputs_embeds
        )

        return text_embeds + audio_embeds

    def process_all_input(self, batch):
        inputs_embeds_list = []
        labels_list = []
        for i in range(len(batch["text"])):
            item = {
                "text": batch["text"][i],
                "audios": batch["audios"][i],
                "audio_lengths": batch["audio_lengths"][0][i].item(),
            }
            (text_modality_tokens, audio_modality_tokens, labels) = (
                self.process_inputs_cls.get_input_label(
                    item, self.codec_model, self.codec_model.device
                )
            )

            labels_list.append(labels)
            input_embeds = self.get_embeds_from_inputs_ids(
                text_modality_tokens, audio_modality_tokens
            )
            inputs_embeds_list.append(input_embeds)

        inputs_embeds = pad_sequence(inputs_embeds_list, batch_first=True, padding_value=0)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=SOFTMAX_IGNORE_INDEX
        )

        return inputs_embeds, labels[:, :, 0], labels[:, :, 1:]

    def _step(self, batch, batch_idx, stage="train"):
        is_train = stage == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # start_time = time()

        inputs_embeds, text_labels, audio_labels = self.process_all_input(batch)

        # log.info(
        #     f"stage: {stage} / get_input_label time: {time() - start_time}"
        # )  # time for codec.encode

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
        
        topk_list = [1, 2, 5, 10, 20]
        # 忽略mambaout_token_id 和 ignore_index
        accuracy_list = self.get_accuracy(multimodel_causual_output.audio_logits, audio_labels, ignore_index=[SOFTMAX_IGNORE_INDEX, self.slow_lm_config.slow_audio_modality_mambaout_token_id], topk=topk_list)
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

        return multimodel_causual_output.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is None:
            for name in list(checkpoint["state_dict"].keys()):
                if "codec" in name:
                    checkpoint["state_dict"].pop(name)
        else:
            # Save only LoRA parameters
            state_dict = checkpoint["state_dict"]
            for name in list(state_dict.keys()):
                if "lora" not in name:
                    state_dict.pop(name)



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
        text_logits = self.text_inference_input_processor(inference_config)
        self.text_prompt_length = text_logits.shape[1]
        self.audio_prompt_length = 0
        self.text_logits = text_logits  # shape = (1, T)

        self.audio_logits = None  # shape = (codebook_count, T)
        self.codebook_shift = (
            torch.arange(self.config.audio_codebook_count)
            * self.config.audio_codebook_size
        ).to(self.model.device)
        input_ids = self.process_2d_logits(
            text_logits, self.audio_logits, mode="infer"
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

    def predict_one_token(
        self, input_ids, past_key_values, inference_config, previous_token=None
    ):
        text_logits, audio_logits, past_kv = self.model.forward_generate(
            input_ids=input_ids, use_cache=True, past_key_values=past_key_values
        )

        next_token_list = [
            self.sample(
                text_logits,
                previous_token=None,
                temperature=inference_config.temperature,
                top_k=inference_config.top_k,
                top_p=inference_config.top_p,
                repetition_penalty=inference_config.windows_penalty,
            )[0]
        ]

        for i in range(self.config.audio_codebook_count):
            next_token_list.append(
                self.sample(
                    audio_logits[:, :, i, :],
                    previous_token=(
                        previous_token[i + 1, :] if previous_token != None else None
                    ),
                    temperature=inference_config.temperature,
                    top_k=inference_config.top_k,
                    top_p=inference_config.top_p,
                    repetition_penalty=inference_config.windows_penalty,
                )[0]
            )

        return torch.stack(next_token_list, dim=0), past_kv

    def prefilling_next_token(
        self, input_ids, past_key_values, previous_token, inference_config
    ):
        return self.predict_one_token(
            input_ids=input_ids,
            past_key_values=past_key_values,
            previous_token=previous_token,
            inference_config=inference_config,
        )

    def predict_n_token(self, input_ids, past_kv, inference_config):
        cur_generation_token_nums = 1
        cur_token = input_ids[:, -1, :].squeeze(0).clone()
        cur_token[1:] = cur_token[1:] + self.codebook_shift
        previous_tokens = torch.zeros(
            (self.config.audio_codebook_count + 1, self.max_length),
            dtype=torch.long,
            device=cur_token.device,
        )

        previous_tokens[:, 0] = cur_token.clone()
        input_ids_ar = input_ids
        past_key_values = past_kv
        while self.is_end_of_predict(cur_generation_token_nums) == False:
            win_size = inference_config.windows_length
            if cur_generation_token_nums < win_size:
                window = previous_tokens[:, :win_size]
            else:
                window = previous_tokens[
                    :, cur_generation_token_nums - win_size : cur_generation_token_nums
                ]
            next_token, past_kv = self.predict_one_token(
                input_ids=input_ids_ar,
                past_key_values=past_key_values,
                inference_config=inference_config,
                previous_token=window,
            )

            cur_generation_token_nums += 1
            previous_tokens[:, cur_generation_token_nums - 1] = next_token.squeeze(
                1
            ).clone()
            input_ids_ar = self.process_input_ids_inference(next_token.squeeze(1))
            past_key_values = past_kv

    def is_end_of_predict(self, cur_generation_token_nums):
        # return (self.text_logits[:, -1].item() == self.config.end_of_music_id) or (cur_generation_token_nums >= self.max_length) or (self.text_logits[:, -1].item() == self.config.end_of_robot_id)
        return cur_generation_token_nums >= self.max_length

    def text_inference_input_processor(self, inference_config):
        text_logits = self.text_tokenizer(inference_config.prompt, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device)

        # text_special_token_start = torch.tensor([self.config.start_of_human_id, self.config.bos_token_id],
        #                                              dtype = torch.long).to(self.model.device)

        # text_special_token_middle_list = torch.tensor([self.config.eos_token_id,
        #                                   self.config.end_of_human_id,
        #                                   self.config.start_of_robot_id,
        #                                   self.config.start_of_music_id], dtype=torch.long).to(self.model.device)

        # input_ids = torch.cat([text_special_token_start, text_logits, text_special_token_middle_list], dim=0)
        return text_logits

    def process_input_ids_inference(self, next_token):
        self.text_logits = (
            torch.cat(
                [self.text_logits, next_token[0].unsqueeze(0).unsqueeze(0)], dim=1
            )
            if self.text_logits is not None
            else next_token[0].unsqueeze(0).unsqueeze(0)
        )
        next_token[1:] = next_token[1:] - self.codebook_shift
        self.audio_logits = (
            torch.cat(
                [self.audio_logits, next_token[1:].unsqueeze(0).T], dim=1
            ).squeeze(0)
            if self.audio_logits is not None
            else next_token[1:].unsqueeze(0).T
        )
        input_ids = self.process_2d_logits(
            self.text_logits if self.text_prompt_length != 0 else None,
            self.audio_logits,
            mode="infer",
        ).T.unsqueeze(0)
        return input_ids

    def sample(
        self,
        logits,
        previous_token: Optional[torch.Tensor] = None,
        temperature=0.7,
        top_k=50,
        top_p=0.7,
        repetition_penalty=1.2,
    ):
        probs = self.logits_to_probs(
            logits[0, -1], previous_token, temperature, top_k, top_p, repetition_penalty
        )

        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def logits_to_probs(
        self,
        logits: torch.Tensor,
        previous_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        # Apply repetition penalty
        if previous_tokens is not None:
            previous_tokens = previous_tokens.long()
            score = torch.gather(logits, dim=-1, index=previous_tokens)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            logits.scatter_(dim=-1, index=previous_tokens, src=score)

        # Step 1: Apply top-k filtering
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k_logits = top_k_logits[..., -1, None]
            logits = torch.where(
                logits < min_top_k_logits,
                torch.full_like(logits, -float("Inf")),
                logits,
            )

        # Step 2: Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # keep at least one option
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        logits = logits / max(temperature, 1e-5)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def multinomial_sample_one_no_sync(
        self,
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        # 确保传入的是一个概率分布
        assert torch.all(probs_sort >= 0), "Probabilities must be non-negative"
        assert torch.isclose(
            probs_sort.sum(), torch.tensor(1.0)
        ), "Probabilities must sum to 1"

        # # 使用 torch.multinomial 进行采样
        idx_next = torch.multinomial(probs_sort, 1)
        return idx_next.to(dtype=torch.long)

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