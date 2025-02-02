import torch


SOFTMAX_IGNORE_INDEX = -100
TWO_SIDE_SILENCE_LENGTH = 6


class ProcessInputs:
    def __init__(self, config, max_length):
        self.config = config
        self.silence_length = config.silence_length

        self.max_length = max_length

    def get_input_label(self, batch, codec_model, device):
        # text_logits_tensor = torch.tensor([batch['text_logits'] for batch in batches], dtype=torch.long).to(self.codec_model.device)
 
        text_logits = batch["text_logits"]
        wav = batch["wav"]
        wav = wav.to(device)
        audio_logits = codec_model.encode(
            wav.unsqueeze(0),
            torch.tensor(wav.shape[-1], device=device).reshape(
                -1,
            ),
        )[0].squeeze(0)  # shape(modality_channels, T)
        if audio_logits.shape[-1] > self.max_length:
            audio_logits = audio_logits[:, : self.max_length]
        # audio_logits = self.logits_shift(audio_logits.squeeze(0))
        text_modality_tokens, audio_modality_tokens, labels = (
            self.process_2d_logits_train(text_logits, audio_logits, device = device)
        )

        return text_modality_tokens, audio_modality_tokens, labels
    

    def process_2d_logits_train(self, text_logits=None, audio_logits=None, device=None):
        # text input_ids create  <SOH><SOT><TOK>...<TOK><EOT><EOH><SOR><SOM>..................text_mamba_out...........<EOM><EOR>
        # audio input_ids create ......................audio_mamba_out......<SLC><SLC><SLC><ATK>...<ATK><SLC><SLC><SLC>...a_m...
        assert text_logits is not None
        assert audio_logits is not None
        assert device is not None
        
        text_length = text_logits.shape[-1]
        audio_length = audio_logits.shape[-1]
        # input_logits = torch.zeros(size=(self.config.audio_codebook_count + 1, text_length + audio_length + self.silence_length*2 + 8), dtype=torch.long)
        labels = torch.full(
            size=(
                self.config.audio_codebook_count + 1,
                text_length + audio_length + self.silence_length * 2 + 8,
            ),
            fill_value=SOFTMAX_IGNORE_INDEX,
            dtype=torch.long,
        ).to(device)
        (
            text_special_token_start_tensor,
            text_special_token_middle_tensor,
            text_special_token_end_tensor,
            text_pad_tokens,
        ) = self.get_text_special_token_start_middle_end(audio_length, device)

        text_modality_tokens = [
            text_special_token_start_tensor,
            (
                text_logits.squeeze(0)
                if text_logits.dim() == 2
                else text_logits
            ),
            text_special_token_middle_tensor,
            text_pad_tokens,
            text_special_token_end_tensor,
        ]

        text_labels = (
            torch.cat(
                [
                    text_special_token_start_tensor,
                    (
                        text_logits.squeeze(0)
                        if text_logits.dim() == 2
                        else text_logits
                    ),
                    text_special_token_middle_tensor,
                    text_pad_tokens,
                    text_special_token_end_tensor,
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(device)
        )

        audio_pad_list = [
            self.config.audio_modality_manbaout_token_id
            for i in range(self.config.audio_codebook_count)
        ]
        # audio_pad_labels_list = [
        #     -100 for i in range(self.config.audio_codebook_count)
        # ]

        audio_start_pad_tokens = torch.tensor(
            [audio_pad_list for _ in range(TWO_SIDE_SILENCE_LENGTH + text_length)], dtype=torch.long
        ).to(device)
        audio_start_pad_labels = audio_start_pad_tokens.clone()

        audio_silence_tokens = torch.tensor(
            [self.config.audio_silence_id for _ in range(self.silence_length)],
            dtype=torch.long,
        ).to(device)

        audio_end_pad_tokens = torch.tensor(
            [audio_pad_list, audio_pad_list], dtype=torch.long
        ).to(device)
        audio_end_pad_labels = audio_end_pad_tokens.clone()
        
        audio_labels = torch.cat(
            [
                audio_start_pad_labels,
                audio_silence_tokens.clone(),
                audio_logits.T.clone(),
                audio_silence_tokens.clone(),
                audio_end_pad_labels,
            ],
            dim=0,
        ).T.to(device)
        
        audio_silence_tokens = self.logits_shift(audio_silence_tokens.T, device=device)
        audio_logits = self.logits_shift(audio_logits, device=device)

        audio_modality_tokens = [
            audio_start_pad_tokens,
            audio_silence_tokens.T,
            audio_logits.T,
            audio_silence_tokens.T,
            audio_end_pad_tokens,
        ]


        labels[0, :] = text_labels
        labels[1:, :] = audio_labels

        return text_modality_tokens, audio_modality_tokens, labels


    def process_2d_logits_infer(self, 
                                device, 
                                text_logits=None, 
                                audio_logits=None,
                                audio_prompt_length=0,
                                text_prompt_length=0):
        if audio_prompt_length == 0 and text_prompt_length > 0:
            text_logits = text_logits[:, : text_prompt_length]
            audio_length = audio_logits.shape[-1] if audio_logits is not None else 0

        if text_logits == None:
            assert audio_logits is not None
            audio_length = audio_logits.shape[-1]

        # text input_ids create <SOH><SOT><Text>...<Text><EOT><EOH><SOR><SOM>...<EOM><EOR>
        text_modality_tokens = None
        (
            text_special_token_start_tensor,
            text_special_token_middle_tensor,
            _,
            text_pad_tokens,
        ) = self.get_text_special_token_start_middle_end(audio_length, device)

        # Text prompt
        if text_prompt_length > 0:
            audio_pad_list = [
                self.config.audio_modality_manbaout_token_id
                for i in range(self.config.audio_codebook_count)
            ]
            audio_start_pad_tokens = torch.tensor(
                [audio_pad_list for _ in range(TWO_SIDE_SILENCE_LENGTH + text_prompt_length)],
                dtype=torch.long,
            ).to(device)
            if audio_length > 0:
                text_modality_tokens = (
                    torch.cat(
                        [
                            text_special_token_start_tensor,
                            (
                                text_logits.squeeze(0)
                                if text_logits.dim() == 2
                                else text_logits
                            ),
                            text_special_token_middle_tensor,
                            text_pad_tokens[self.silence_length * 2 :],
                        ],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                audio_modality_tokens = torch.cat(
                    [audio_start_pad_tokens, audio_logits.T], dim=0
                ).to(device)

            elif audio_length == 0:
                text_modality_tokens = (
                    torch.cat(
                        [
                            text_special_token_start_tensor,
                            (
                                text_logits.squeeze(0)
                                if text_logits.dim() == 2
                                else text_logits
                            ),
                            text_special_token_middle_tensor,
                        ],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                audio_modality_tokens = audio_start_pad_tokens
            return torch.cat(
                [text_modality_tokens, audio_modality_tokens.T], dim=0
            ).to(device)

        # Audio prompt
        if text_prompt_length == 0:
            text_modality_tokens = text_pad_tokens.unsqueeze(0)[:, :-3].to(
                device
            )
            audio_silence_tokens = torch.tensor(
                [self.config.audio_silence_id for i in range(self.silence_length)],
                dtype=torch.long,
            ).to(device)
            audio_modality_tokens = torch.cat(
                [audio_silence_tokens, audio_logits.T], dim=0
            ).to(device)
            return torch.cat(
                [text_modality_tokens, audio_modality_tokens.T], dim=0
            ).to(device)

    def get_text_special_token_start_middle_end(self, audio_length, device):
        text_special_token_start_list = []
        text_special_token_start_list.append(self.config.start_of_human_id)
        text_special_token_start_list.append(self.config.bos_token_id)
        text_special_token_start_tensor = torch.tensor(
            text_special_token_start_list, dtype=torch.long
        ).to(device)

        text_special_token_middle_list = []
        text_special_token_middle_list.append(self.config.eos_token_id)
        text_special_token_middle_list.append(self.config.end_of_human_id)
        text_special_token_middle_list.append(self.config.start_of_robot_id)
        text_special_token_middle_list.append(self.config.start_of_music_id)
        text_special_token_middle_tensor = torch.tensor(
            text_special_token_middle_list, dtype=torch.long
        ).to(device)

        text_special_token_end_list = []
        text_special_token_end_list.append(self.config.end_of_music_id)
        text_special_token_end_list.append(self.config.end_of_robot_id)
        text_special_token_end_tensor = torch.tensor(
            text_special_token_end_list, dtype=torch.long
        ).to(device)

        if audio_length > 0:
            text_pad_tokens = [
                self.config.text_modality_manbaout_token_id
                for i in range(self.silence_length * 2 + audio_length)
            ]
            text_pad_tokens = torch.tensor(text_pad_tokens, dtype=torch.long).to(
                device
            )
        else:
            text_pad_tokens = None

        return (
            text_special_token_start_tensor,
            text_special_token_middle_tensor,
            text_special_token_end_tensor,
            text_pad_tokens,
        )
    
    def logits_shift(self, audio_logits, device):
        """
            labels.shape=[modality, :]
        """
        
        audio_logits_shift = (
            torch.arange(self.config.audio_codebook_count, device=audio_logits.device)
            * self.config.audio_codebook_size
        ).to(device)
        
        audio_logits += audio_logits_shift.unsqueeze(-1)
        
        return audio_logits