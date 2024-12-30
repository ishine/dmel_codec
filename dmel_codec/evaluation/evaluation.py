import torch
import numpy as np
from dmel_codec.evaluation.initial_codec import InitialCodec
from dmel_codec.evaluation.evaluation_utils import (
    wer,
    calculate_spk_sim,
    calculate_stoi,
    calculate_pesq,
)
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from dmel_codec.lhotse_tts_dataset import LhotseDataModule


class Evaluation:
    def __init__(
        self,
        codec_name: str,
        ckpt_path: str = None,
        device: str = "cpu",
        sample_rate: int = 24000,
        num_quantizers: int | None = None,
        asr_model_ckpt_path: str = None,
        spk_embedding_model_ckpt_path: str = None,
        max_duration: int = 60,  # a batch max duration
        test_recordings_paths: str | None | list[str] = None,
        test_supervisions_paths: str | None | list[str] = None,
        test_cuts_paths: str | None | list[str] = None,
        test_prefix: str | None | list[str] = None,
        test_max_samples: int | None = None,
        stage: str = "test",
    ):
        self.codec = InitialCodec(
            codec_name, ckpt_path, device, sample_rate, num_quantizers
        )
        self.spk_embedding_model = EncoderClassifier.from_hparams(
            source=spk_embedding_model_ckpt_path
        )
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model_ckpt_path)
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_ckpt_path)
        self.max_duration = max_duration

        self.test_dataloader = self.initial_dataloader(
            test_recordings_paths,
            test_supervisions_paths,
            test_cuts_paths,
            test_prefix,
            test_max_samples,
            stage,
        )

    def initial_dataloader(
        self,
        test_recordings_paths,
        test_supervisions_paths,
        test_cuts_paths,
        test_prefix,
        test_max_samples,
        stage,
    ):
        data_module = LhotseDataModule(
            test_recordings_paths=test_recordings_paths,
            test_supervisions_paths=test_supervisions_paths,
            test_cuts_paths=test_cuts_paths,
            test_prefix=test_prefix,
            test_max_samples=test_max_samples,
            stage=stage,
        )
        data_module.setup("test")
        test_dataloader = data_module.test_dataloader()
        return test_dataloader

    @torch.no_grad()
    def get_wer(self, rec_audio, g_audio, gt_text):
        return wer(
            rec_audio,
            g_audio,
            gt_text,
            self.asr_processor,
            self.asr_model,
            sample_rate=self.codec.sample_rate,
        )

    @torch.no_grad()
    def get_pesq(self, rec_audio, g_audio):
        return calculate_pesq(rec_audio, g_audio, sample_rate=self.codec.sample_rate)

    @torch.no_grad()
    def get_stoi(self, rec_audio, g_audio):
        return calculate_stoi(rec_audio, g_audio, sample_rate=self.codec.sample_rate)

    @torch.no_grad()
    def get_spk_sim(self, rec_audio, g_audio):
        return calculate_spk_sim(
            g_audio,
            rec_audio,
            self.spk_embedding_model,
            sample_rate=self.codec.sample_rate,
        )

    def evaluation(self):
        wer_list = []
        pesq_list = []
        stoi_list = []
        spk_sim_list = []
        for batch in self.test_dataloader:
            wer, pesq, stoi, spk_sim = self.step(batch)
            wer_list.append(wer)
            pesq_list.append(pesq)
            stoi_list.append(stoi)
            spk_sim_list.append(spk_sim)
        return {
            "wer": np.mean(np.array(wer_list)),
            "pesq": np.mean(np.array(pesq_list)),
            "stoi": np.mean(np.array(stoi_list)),
            "spk_sim": np.mean(np.array(spk_sim_list)),
        }

    def step(self, batch):
        text = batch["text"]
        gt_audio = batch["audios"]
        gt_audio_for_rec = gt_audio.clone()
        duration = batch["audio_lengths"]

        rec_audio = self.codec.rec_audio(gt_audio_for_rec, duration)

        wer = self.get_wer(rec_audio, gt_audio, text)
        pesq = self.get_pesq(rec_audio, gt_audio)
        stoi = self.get_stoi(rec_audio, gt_audio)
        spk_sim = self.get_spk_sim(rec_audio, gt_audio)

        return wer, pesq, stoi, spk_sim

    # def get_visqol(self, rec_audio, g_audio):
    #     pass

if __name__ == "__main__":
    evaluation = Evaluation()
    evaluation.evaluation()
