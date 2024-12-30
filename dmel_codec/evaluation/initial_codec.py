from pathlib import Path
import hydra
from dmel_codec.models.lit_modules import VQGAN
import torch


# initial speechtokenizer | DAC | Encodec | Ours_dMel_codec | mimi(moshi codec)
class InitialCodec:
    def __init__(
        self,
        codec_name: str,
        ckpt_path: str = None,
        device: str = "cpu",
        sample_rate: int = 24000,
        num_quantizers: int | None = None,
    ):
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
            self.codec.load_state_dict(torch.load(self.ckpt_path)["state_dict"], strict=False) # vocoder have loaded in init

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
            assert self.ckpt_path is not None, "ckpt_path must be provided for mimi codec"
        
        if self.num_quantizers is None:
            print("No num_quantizers provided, using default quantizers")
        else:
            print(f"Using {self.num_quantizers} quantizers")

    @torch.no_grad()
    @torch.inference_mode()
    def extract_codes(self, audios: torch.Tensor, audio_lens: torch.Tensor):
        # audios.shape = (batch_size, 1, audio_len)
        # audio_lens.shape = (batch_size,)
        feature_lens = None
        if self.codec_name == "dMel":
            indices, feature_lens = self.codec.encode(audios, audio_lens)
        else:
            pass

        return indices, feature_lens

    @torch.no_grad()
    @torch.inference_mode()
    def extract_latent_unquantized(self, audios: torch.Tensor, audio_lens: torch.Tensor):
        if self.codec_name == "dMel":
            unquantized_features, mel_lengths = self.codec.encode_unquantized(audios, audio_lens)
            return unquantized_features, mel_lengths
        else:
            pass

    @torch.no_grad()
    @torch.inference_mode()
    def extract_latent_quantized(self, audios: torch.Tensor, audio_lens: torch.Tensor):
        if self.codec_name == "dMel":
            unquantized_features, mel_lengths = self.codec.encode_unquantized(audios, audio_lens)
            indices, indices_lengths = self.codec.get_indices_from_unquantized_features(unquantized_features, mel_lengths)
            quantized_features, mel_masks_float_conv = self.codec.get_quantized_features_from_indices(indices, indices_lengths)
            return quantized_features, mel_masks_float_conv
        else:
            pass

    @torch.no_grad()
    @torch.inference_mode()
    def rec_audio_from_indices(self, indices: torch.Tensor, indices_lengths: torch.Tensor):
        if self.codec_name == "dMel":
            rec_audios, gen_mel = self.codec.decode(indices, indices_lengths, return_audios=True)
            return rec_audios, gen_mel
        else:
            pass

    @torch.no_grad()
    @torch.inference_mode()
    def rec_from_unquantized_latent(self, latent: torch.Tensor, mel_lengths: torch.Tensor):
        if self.codec_name == "dMel":
            indices, indices_lengths = self.codec.get_indices_from_unquantized_features(latent, mel_lengths)
            rec_audios, gen_mel = self.codec.decode(indices, indices_lengths, return_audios=True)
            return rec_audios, gen_mel
        else:
            pass

    @torch.no_grad()
    @torch.inference_mode()
    def rec_from_quantized_latent(self, latent: torch.Tensor, mel_masks_float_conv: torch.Tensor):
        if self.codec_name == "dMel":
            gen_mel = (
                self.codec.decoder(
                    torch.randn_like(latent).to(self.codec.encode_dtype) * mel_masks_float_conv,
                    condition=latent,
                )
                * mel_masks_float_conv
            )
            audios = self.codec.vocoder(gen_mel)
            return audios
        else:
            pass

    @torch.no_grad()
    @torch.inference_mode()
    def rec_audio_from_audio(self, audios: torch.Tensor, audio_lens: torch.Tensor):
        if self.codec_name == "dMel":
            indices, indices_lengths = self.codec.encode(audios, audio_lens)
            rec_audios, gen_mel = self.codec.decode(indices, indices_lengths, return_audios=True)
            return rec_audios, gen_mel
        else:
            pass

    def split_from_length(self, audios: torch.Tensor, length: int):
        # There are some audios that are shorter than the audio_len, we need to split them into the corresponding length
        # audios.shape = (batch_size, 1, audio_len)
        # length.shape = (batch_size,)
        assert audios.shape[0] > 1 and audios.shape[0] == length.shape[0], "batch_size must be greater than 1 and equal to length.shape[0]"
        split_audios = []
        for i in range(audios.shape[0]):
            split_audios.append(audios[i, :, :length[i]])
        return split_audios

if __name__ == "__main__":
    import torchaudio
    from dmel_codec.utils.utils import open_filelist
    initial_codec = InitialCodec(
        codec_name="dMel",
        ckpt_path="/home/wzy/projects/dmel_codec/dmel_codec/ckpt/epoch=018-step=746000_20hz.ckpt",
        device="cpu",
        sample_rate=24000
    )
    
    # load filelist to find audio path
    audio_path_list = open_filelist("/sdb/data1_filelist/filelist_VCTK-Corpus.txt")

    for audio_path in audio_path_list:
        audio, sr = torchaudio.load(audio_path)
        if sr != initial_codec.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, initial_codec.sample_rate)
        audio = audio.unsqueeze(0).to(initial_codec.device)
        audio_lens = torch.tensor([audio.shape[-1]]).to(initial_codec.device)
        rec_audio, _ = initial_codec.rec_audio_from_audio(audio, audio_lens)
        print(rec_audio.shape)