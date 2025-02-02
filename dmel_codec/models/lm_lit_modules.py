import lightning.pytorch as pl



class MusicLLM(pl.LightningModule):
    def __init__(
        self,
        llm_config_path: str,
        optimizer: Any,
        lr_scheduler: Any,
        # Codec
        codec_model: nn.Module,
        # Other parameters
        silence_length: int,
        max_length: int,
        is_strict: bool,
    ):
        super().__init__()
        # 原生最大词表利用到151643, 最大长度是151936

        config = Qwen2Config.from_pretrained(llm_config_path)
        self.config = config
        self.lora_config = config.lora_config
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.strict_loading = is_strict
        model = ChatMusicQwen2ForCausalLM(config)
        if config.qwen2_mllm_from_pretrain is None:
            assert config.qwen2_text_base is not None
            log.info("Loading qwen2 text base model from pretrained checkpoints")
            model.load_state_dict(
                load_file(config.qwen2_text_base, device="cpu"), strict=False
            )
            log.info(f"Loading qwen2 mllm model from {config.qwen2_text_base}")
            zero_tensor = torch.zeros_like(
                model.model.embed_tokens.weight[config.text_modality_manbaout_token_id]
            ).detach()
            model.model.embed_tokens.weight.data[config.text_modality_manbaout_token_id] = zero_tensor
            
            self.model = model
        else:
            log.info("Loading qwen2 mllm model from pretrained checkpoints")
            self.model = model

        self.max_length = max_length
        self.silence_length = silence_length
        # self.codec_model = self.load_codec_model(
        #     codec_experiment=codec_experiment,
        #     codec_vocoder_config_path=codec_vocoder_config_path,
        #     codec_config_dir=codec_config_dir,
        #     codec_ckpt_path=codec_ckpt_path,
        # )
        self.codec_model = codec_model
        self.process_inputs_cls = ProcessInputs(self.config, max_length)

    @staticmethod
    def get_codec_config(
        codec_experiment: str,
        codec_vocoder_config_path: str,
        codec_config_dir: str,
        dtype: str | torch.dtype,
    ):
        cfg = None
        with initialize_config_dir(
            config_dir=codec_config_dir, job_name="load_codec_model"
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={codec_experiment}",
                    "+dmel_vocoder_ckpt_path=null",
                    f"+dmel_vocoder_config_path='{codec_vocoder_config_path}'",
                    f"+model.dtype={dtype}",
                ],
            )
        OmegaConf.resolve(cfg)
        return OmegaConf.to_yaml(cfg)

    @staticmethod
    def load_codec_model(
        codec_experiment: str = "dmel_pertrain_ngroups10_ncodebooks1_levels86_downsample22_encoderresiduallayers16",
        codec_vocoder_config_path: str = "/home/wzy/projects/bigvgan_v2_24khz_100band_256x/config.json",
        codec_config_dir: str = "/home/wzy/projects/zh_dmel_chat_music/dmel_codec/config/",
        codec_ckpt_path: str = "/home/wzy/projects/checkpoints/epoch=243-step=635000_20hz.ckpt",
        dtype: torch.dtype | str = "bfloat16",
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        codec_model = None
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=codec_config_dir, job_name="load_codec_model"
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={codec_experiment}",
                    "+dmel_vocoder_ckpt_path=null",
                    f"+dmel_vocoder_config_path='{codec_vocoder_config_path}'",
                    f"+model.dtype={dtype}",
                ],
            )
            if cfg is None:
                raise ValueError("Codec config is None")
            codec_model = hydra.utils.instantiate(
                cfg.model,
            )

        if codec_model is None:
            raise ValueError("Codec model is None")

        if codec_ckpt_path == None:
            pass
        else:
            codec_model.load_state_dict(
                torch.load(
                    codec_ckpt_path,
                    map_location="cpu",
                )["state_dict"],
                strict=False,
            )
            log.info('codec ckpt read succesfully')
        codec_model.eval()
        for param in codec_model.parameters():
            param.requires_grad = False
        codec_model = codec_model.to(dtype=dtype)
        return codec_model

    def get_accuracy(self, logits, labels):
        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[labels == SOFTMAX_IGNORE_INDEX] = 0
        correct = correct.sum()
        accuracy = correct / (labels != SOFTMAX_IGNORE_INDEX).sum()
        return accuracy

    def configure_optimizers(self) -> OptimizerLRScheduler:
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

    def get_embeds_from_inputs_ids(self, text_logits_list, audio_logits_list):
        """
        text_logits_list: 二维数组
        audio_logits_list: 二维数组
        """
        text_embeds_list = []
        for text_logits in text_logits_list:
            if (text_logits == self.config.text_modality_manbaout_token_id).all():
                with torch.no_grad():
                    text_embeds_list.append(self.model.model.embed_tokens(text_logits))
            else:
                text_embeds_list.append(self.model.model.embed_tokens(text_logits))

        audio_embeds_list = []
        for audio_logits in audio_logits_list:
            if (audio_logits == self.config.text_modality_manbaout_token_id).all():
                with torch.no_grad():
                    audio_embeds_list.append(
                        self.model.model.audio_embeddings(audio_logits)
                    )
            else:
                audio_embeds_list.append(
                    self.model.model.audio_embeddings(audio_logits)
                )

        with torch.no_grad():
            text_embeds = torch.cat(text_embeds_list, dim=0)
            audio_embeds = torch.cat(audio_embeds_list, dim=0)

        audio_embeds = rearrange(
            audio_embeds,
            "s c d -> s (c d)",
            c=self.config.audio_codebook_count,
            d=self.config.audio_token_original_embed_dim,
        )

        audio_embeds = self.model.model.audio_embeddings_projector(audio_embeds)

        return text_embeds + audio_embeds

    def pad_embed_sequence(self, embeds_list, padding_value):
        """
        embeds_list: [embeds1, embeds2]
        embeds1.shape = (seq_len, hidden_state)
        """
        max_len = max(embeds.shape[0] for embeds in embeds_list)

        padded_embeds = torch.full(
            size=(len(embeds_list), max_len, embeds_list[0].shape[-1]),
            fill_value=padding_value,
            dtype=self.model.dtype,
            device=self.model.device,
        )
        for i, embeds in enumerate(embeds_list):
            pad_len = max_len - embeds.shape[0]
            padded_embed = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len))
            padded_embeds[i] = padded_embed

        return padded_embeds

    def process_all_input(self, batch):
        inputs_embeds_list = []
        labels_list = []
        for item in batch:
            (text_modality_tokens, audio_modality_tokens, labels) = (
                self.process_inputs_cls.get_input_label(
                    item, self.codec_model, self.codec_model.device
                )
            )

            labels_list.append(labels.T)
            input_embeds = self.get_embeds_from_inputs_ids(
                text_modality_tokens, audio_modality_tokens
            )
            inputs_embeds_list.append(input_embeds)

        inputs_embeds = self.pad_embed_sequence(inputs_embeds_list, padding_value=0)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=SOFTMAX_IGNORE_INDEX
        )

        return inputs_embeds, labels[:, :, 0], labels[:, :, 1:]

    def _step(self, batch, batch_idx, stage="train"):
        is_train = stage == "train"
        if is_train == "train":
            self.model.train()

        # start_time = time()

        inputs_embeds, text_labels, audio_labels = self.process_all_input(batch)

        # log.info(
        #     f"stage: {stage} / get_input_label time: {time() - start_time}"
        # )  # time for codec.encode

        start_time = time()
        multimodel_causual_output: MultiModalCausalLMOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            text_labels=text_labels,
            audio_labels=audio_labels,
        )
        # log.info(f"stage: {stage} / forward time: {time() - start_time}")

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

        audio_logits = multimodel_causual_output.audio_logits.transpose(1, 2)

        accuracy = self.get_accuracy(audio_logits, audio_labels)
        self.log(
            f"{stage}/top_5_accuracy",
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

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

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

    @torch.no_grad()
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
        text_tokenizer = AutoTokenizer.from_pretrained(
            inference_config.text_tokenizer_path
        )
        text_logits = text_tokenizer(inference_config.prompt, return_tensors="pt")[
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