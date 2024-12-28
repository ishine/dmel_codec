from pathlib import Path
import hydra
from dmel_codec.models.lit_modules import VQGAN
import torch


# initial speechtokenizer | DAC | Encodec | Ours_dMel_codec | mimi(moshi codec)
class InitialCodec:
    def __init__(self, codec_name: str, ckpt_path: str = None, device: str = "cpu", sample_rate: int = 24000, num_quantizers: int | None = None):
        self.codec_name = codec_name
        self.ckpt_path = ckpt_path
        self.device = device
        self.sample_rate = sample_rate
        self.num_quantizers = num_quantizers
        self.hparams_check()
        if self.codec_name == "speechtokenizer":
            from speechtokenizer.model import SpeechTokenizer
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id="fnlp/SpeechTokenizer")
            checkpoints_path = Path(f"{path}/speechtokenizer_hubert_avg")
            self.codec = SpeechTokenizer.load_from_checkpoint(
                checkpoints_path / "config.json",
                checkpoints_path / "SpeechTokenizer.pt",
            )

        elif self.codec_name == "DAC":
            import dac
            model_path = dac.utils.download(model_type=f"{self.sample_rate//1000}khz")
            self.codec = dac.DAC.load(model_path)
    
        elif self.codec_name == "Encodec":
            from transformers import EncodecModel
            self.codec = EncodecModel.from_pretrained(self.ckpt_path)
            
        elif self.codec_name == "dMel":
            from dmel_codec.train import get_config
            config = get_config()
            self.codec = hydra.utils.instantiate(config.model, _convert_="partial")
            self.codec.load_state_dict(torch.load(config.ckpt_path)['state_dict'])

        elif self.codec_name == "mimi":
            from transformers.models.mimi.modeling_mimi import MimiModel
            from transformers.models.mimi.configuration_mimi import MimiConfig
            config = MimiConfig.from_pretrained(self.ckpt_path)
            self.codec = MimiModel.from_pretrained(self.ckpt_path, config=config)

        self.codec.to(self.device)
        self.codec.eval()

    def hparams_check(self):
        assert self.codec_name in [
            "speechtokenizer",
            "DAC",
            "Encodec",
            "dMel",
            "mimi",
        ], "Invalid codec name, assert codec_name in ['speechtokenizer', 'DAC', 'Encodec', 'dMel', 'mimi']"
        if self.codec_name == "dMel":
            assert (
                self.ckpt_path is not None
            ), "ckpt_path must be provided for dMel codec"
        if self.codec_name == "mimi":
            assert self.ckpt_path is None, "ckpt_path must be None for mimi codec"

    def extract_codes(self, audios, audio_lens):
        pass

    def extract_latent_unquantized(self, audios, audio_lens):
        pass

    def extract_latent_quantized(self, audios, audio_lens):
        pass

    def rec_audio(self, audios, audio_lens):
        pass

    def rec_from_unquantized_latent(self, latent):
        pass

    def rec_from_quantized_latent(self, latent):
        pass
