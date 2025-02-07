import torch
from dmel_codec.models.modules.config_lm import Qwen2Config
from transformers import AutoTokenizer
SOFTMAX_IGNORE_INDEX = -100
TEXT_SPECIAL_TOKEN_LENGTH = 8


class ProcessInputs:
    def __init__(
        self,
        config: Qwen2Config,
        max_length: int,
        silence_length: int,
        audio_silence_id: list[int],
        text_tokenizer: AutoTokenizer,
    ):
        """
            Process inputs for LM
            config: slow_lm_config
            max_length: max_token_length
            silence_length: silence seconds
            audio_silence_id: audio silence ids for one frame 
        """
        self.config = config
        self.silence_length = silence_length
        self.max_length = max_length
        self.audio_silence_id = audio_silence_id
        self.text_tokenizer = text_tokenizer
    
    def get_audio_logits_parralel(self, wav, audio_lengths, codec_model):
        with torch.no_grad():
            audio_logits, logits_lengths = codec_model.encode(
                wav,
                audio_lengths
            )
            # garatee batch size 1 can be processed
            if audio_logits.dim() == 2:
                audio_logits = audio_logits.unsqueeze(0)

            audio_logits_list = []
            logits_lengths = logits_lengths.view(-1)
            for i in range(audio_logits.shape[0]):
                if logits_lengths[i].item() > self.max_length:
                    audio_logits_tmp = audio_logits[i, :, :self.max_length]
                else:
                    audio_logits_tmp = audio_logits[i, :, :logits_lengths[i].item()]
                audio_logits_list.append(audio_logits_tmp.T) # shape(T, codebook_num)

            return audio_logits_list



    def get_input_label(self, item, device):
        # input one batch

        text = item["text"]
        audio_logits = item["audio_logits"]

        text_logits = self.text_tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0).to(device)


        text_modality_tokens, audio_modality_tokens, labels = (
            self.process_2d_logits_train(text_logits, audio_logits, device=device)
        )

        return text_modality_tokens, audio_modality_tokens, labels

    def process_2d_logits_train(self, text_logits=None, audio_logits=None, device=None):
        """
            Process 2d logits for LM
            text_logits: text logits (text_length)
            audio_logits: audio logits (audio_length, codebook_num)
            device: device
            
            return:
                text_modality_tokens: [T]
                audio_modality_tokens: [T, codebook_num]
                labels: [T, codebook_num + 1]
        """
        # text input_ids create  <SOH><SOT><TOK>...<TOK><EOT><EOH><SOR><SOM>..................text_mamba_out...........<EOM><EOR>
        # audio input_ids create ......................audio_mamba_out......<SLT><SLT><SLT><ATK>...<ATK><SLT><SLT><SLT>...a_m...
        assert text_logits is not None
        assert audio_logits is not None
        assert device is not None

        text_length = text_logits.shape[0]
        audio_length = audio_logits.shape[0]
        # input_logits = torch.zeros(size=(self.config.audio_codebook_count + 1, text_length + audio_length + self.silence_length*2 + 8), dtype=torch.long)
        labels = torch.full(
            size=(
                text_length + audio_length + self.silence_length * 2 + TEXT_SPECIAL_TOKEN_LENGTH,
                self.config.audio_codebook_count + 1
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

        text_modality_tokens = torch.cat([
            text_special_token_start_tensor,
            (text_logits.squeeze(0) if text_logits.dim() == 2 else text_logits),
            text_special_token_middle_tensor,
            text_pad_tokens,
            text_special_token_end_tensor,
        ], dim=0).to(device)

        audio_pad_list = [
            self.config.slow_audio_modality_mambaout_token_id
            for _ in range(self.config.audio_codebook_count)
        ]

        audio_start_pad_tokens = torch.tensor(
            [audio_pad_list for _ in range(TEXT_SPECIAL_TOKEN_LENGTH + text_length - 2)], # -2 for <EOM> and <EOR>
            dtype=torch.long,
        ).to(device)

        audio_silence_tokens = torch.tensor(
            [self.audio_silence_id for _ in range(self.silence_length)],
            dtype=torch.long,
        ).to(device)

        audio_end_pad_tokens = torch.tensor(
            [audio_pad_list, audio_pad_list], dtype=torch.long # 2 audio_pad_list for <EOM> and <EOR>
        ).to(device)

        audio_silence_tokens = self.logits_shift(audio_silence_tokens, device=device)
        audio_logits = self.logits_shift(audio_logits, device=device)

        audio_modality_tokens = torch.cat([
            audio_start_pad_tokens,
            audio_silence_tokens,
            audio_logits,
            audio_silence_tokens,
            audio_end_pad_tokens,
        ], dim=0).to(device)

        # 模态让位符也需要更新梯度, 所以需要将text_modality_tokens和audio_modality_tokens赋值给labels
        labels[:, 0] = text_modality_tokens
        labels[:, 1:] = audio_modality_tokens

        return text_modality_tokens, audio_modality_tokens, labels

    def process_2d_logits_infer(
        self,
        device,
        text_logits=None,
        audio_logits=None,
        audio_prompt_length=0,
        text_prompt_length=0,
    ):
        if audio_prompt_length == 0 and text_prompt_length > 0:
            text_logits = text_logits[:, :text_prompt_length]
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
                self.config.audio_modality_mambaout_token_id
                for i in range(self.config.audio_codebook_count)
            ]
            audio_start_pad_tokens = torch.tensor(
                [
                    audio_pad_list
                    for _ in range(TEXT_SPECIAL_TOKEN_LENGTH + text_prompt_length - 2) # -2 代表text token的最后有两个special token
                ],
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
            return torch.cat([text_modality_tokens, audio_modality_tokens.T], dim=0).to(
                device
            )

        # Audio prompt
        if text_prompt_length == 0:
            text_modality_tokens = text_pad_tokens.unsqueeze(0)[:, :-3].to(device)
            audio_silence_tokens = torch.tensor(
                [self.audio_silence_id for i in range(self.silence_length)],
                dtype=torch.long,
            ).to(device)
            audio_modality_tokens = torch.cat(
                [audio_silence_tokens, audio_logits.T], dim=0
            ).to(device)
            return torch.cat([text_modality_tokens, audio_modality_tokens.T], dim=0).to(
                device
            )

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
                self.config.text_modality_mambaout_token_id
                for i in range(self.silence_length * 2 + audio_length)
            ]
            text_pad_tokens = torch.tensor(text_pad_tokens, dtype=torch.long).to(device)
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

        audio_logits += audio_logits_shift.unsqueeze(0)

        return audio_logits


if __name__ == "__main__":
    from transformers.models.qwen2 import Qwen2Tokenizer
    device = "cpu"
    slow_lm_config = Qwen2Config.from_pretrained(
        "/home/wzy/projects/dmel_codec/dmel_codec/config/lm/slow_lm_0.5B.json"
    )
    processer = ProcessInputs(
        config=slow_lm_config,
        max_length=4096,
        silence_length=3,
        audio_silence_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        text_tokenizer=Qwen2Tokenizer.from_pretrained("/sdb/model_weight/qwen2-0.5B"),
    )
    fake_text_inputs_ids = torch.randint(0, 151936, size=(3, 20)).to(device)

    fake_audio_list = []
    for i in range(10):
        fake_audio_list.append(torch.randint(175 * i, 175 * (i + 1), (3, 20)))
    fake_audio_inputs_ids = torch.stack(fake_audio_list, dim=2).to(device)

    for i in range(fake_text_inputs_ids.shape[0]):
        text_modality_tokens, audio_modality_tokens, labels = processer.process_2d_logits_train(
            fake_text_inputs_ids[i], fake_audio_inputs_ids[i], device=device
        )
        print(text_modality_tokens.shape)
        print(audio_modality_tokens.shape)
        print(labels.shape)
        print((text_modality_tokens == labels[:, 0]).all())
        print((audio_modality_tokens == labels[:, 1:]).all())
        print(f'------------{i}----------')