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
from tqdm import tqdm

class Evaluation:
    def __init__(
        self,
        codec_name: str,
        ckpt_path: str = None,
        device: str = "cpu",
        sample_rate: int = 24000,
        num_quantizers: int | None = None,
        asr_model_ckpt_path: str = None,
        spk_embedding_model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        max_duration: int = 60,  # a batch max duration
        test_recordings_paths: str | None | list[str] = None,
        test_supervisions_paths: str | None | list[str] = None,
        test_cuts_paths: str | None | list[str] = None,
        test_prefix: str | None | list[str] = None,
        test_max_samples: int | None = None,
        stage: str = "test",
        test_num_workers: int = 4,
    ):
        self.codec = InitialCodec(
            codec_name, ckpt_path, device, sample_rate, num_quantizers
        )
        
        self.spk_embedding_model = EncoderClassifier.from_hparams(
            source=spk_embedding_model_name,
        )
        self.spk_embedding_model.eval()
        self.spk_embedding_model.device = device
        self.spk_embedding_model.mods.to(device)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_model_ckpt_path)
        self.asr_processor = AutoProcessor.from_pretrained(asr_model_ckpt_path)
        self.asr_model.to(device)
        self.max_duration = max_duration
        self.device = device

        self.test_dataloader = self.initial_dataloader(
            test_recordings_paths,
            test_supervisions_paths,
            test_cuts_paths,
            test_prefix,
            test_max_samples,
            stage,
            test_num_workers,
        )

    def initial_dataloader(
        self,
        test_recordings_paths,
        test_supervisions_paths,
        test_cuts_paths,
        test_prefix,
        test_max_samples,
        stage,
        test_num_workers,
    ):
        data_module = LhotseDataModule(
            test_recordings_paths=test_recordings_paths,
            test_supervisions_paths=test_supervisions_paths,
            test_cuts_paths=test_cuts_paths,
            test_prefix=test_prefix,
            test_max_samples=test_max_samples,
            test_max_duration=self.max_duration,
            stage=stage,
            test_num_workers=test_num_workers,
        )
        data_module.setup("test")
        test_dataloader = data_module.test_dataloader()
        return test_dataloader

    @torch.inference_mode()
    def get_wer(self, rec_audio, gt_audio, gt_text):
        return wer(
            gt_audio,
            rec_audio,
            gt_text,
            self.asr_processor,
            self.asr_model,
            sample_rate=self.codec.sample_rate,
        )

    @torch.inference_mode()
    def get_pesq(self, rec_audio, g_audio):
        return calculate_pesq(rec_audio, g_audio, sample_rate=self.codec.sample_rate)

    @torch.inference_mode()
    def get_stoi(self, rec_audio, g_audio):
        return calculate_stoi(rec_audio, g_audio, sample_rate=self.codec.sample_rate)

    @torch.inference_mode()
    def get_spk_sim(self, rec_audio, g_audio):
        return calculate_spk_sim(
            g_audio,
            rec_audio,
            self.spk_embedding_model,
            sample_rate=self.codec.sample_rate,
        )

    def evaluation(self):
        wer_gt_list = []
        wer_rec_list = []
        pesq_list = []
        stoi_list = []
        spk_sim_list = []
        for batch in tqdm(self.test_dataloader):
            wer_gt, wer_rec, pesq, stoi, spk_sim = self.step(batch)
            print(f"wer_gt: {wer_gt}, wer_rec: {wer_rec}, pesq: {pesq}, stoi: {stoi}, spk_sim: {spk_sim}")
            wer_gt_list.append(wer_gt)
            wer_rec_list.append(wer_rec)
            pesq_list.append(pesq)
            stoi_list.append(stoi)
            spk_sim_list.append(spk_sim)
        return {
            "wer_gt": np.mean(np.array(wer_gt_list)),
            "wer_rec": np.mean(np.array(wer_rec_list)),
            "pesq": np.mean(np.array(pesq_list)),
            "stoi": np.mean(np.array(stoi_list)),
            "spk_sim": np.mean(np.array(spk_sim_list)),
        }

    def step(self, batch):
        text = batch["text"]
        text = [t[0] for t in text]
        gt_audio = batch["audios"]
        if gt_audio.ndim == 2:
            gt_audio = gt_audio.unsqueeze(1).to(self.device)
        else:
            gt_audio = gt_audio.transpose(1, 0).to(self.device)
        print(f"gt_audio shape: {gt_audio.shape}")
        gt_audio_for_rec = gt_audio.clone()
        audio_lengths = batch["audio_lengths"]

        rec_audio, _ = self.codec.rec_audio_from_audio(gt_audio_for_rec, audio_lengths)

        if rec_audio.shape[-1] > gt_audio.shape[-1]:
            rec_audio = rec_audio[:, :, :gt_audio.shape[-1]]
        else:
            gt_audio = gt_audio[:, :, :rec_audio.shape[-1]]

        wer_gt, wer_rec = self.get_wer(rec_audio, gt_audio, text)
        pesq = self.get_pesq(rec_audio, gt_audio)
        stoi = self.get_stoi(rec_audio, gt_audio)
        spk_sim = self.get_spk_sim(rec_audio, gt_audio)

        return wer_gt, wer_rec, pesq, stoi, spk_sim

    # def get_visqol(self, rec_audio, g_audio):
    #     pass

if __name__ == "__main__":
    evaluation = Evaluation(
        codec_name="speechtokenizer",
        # ckpt_path="/home/wzy/projects/dmel_codec/dmel_codec/ckpt/dmel_codec/epoch=022-step=906000_20hz.ckpt",
        ckpt_path="/sdb/model_weight/speechTokenizer/speechtokenizer_hubert_avg",
        # ckpt_path=None,
        device="cuda:1",
        sample_rate=24000,
        num_quantizers=8,
        asr_model_ckpt_path='/sdb/model_weight/whisper-base',
        spk_embedding_model_name='speechbrain/spkrec-ecapa-voxceleb',
        max_duration=150,
        test_recordings_paths=[
            "/sdb/data1/lhotse/libritts/libritts_recordings_test-clean.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_recordings_test-other.jsonl.gz",
        ],
        test_supervisions_paths=[
            "/sdb/data1/lhotse/libritts/libritts_supervisions_test-clean.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_supervisions_test-other.jsonl.gz",
        ],
        test_cuts_paths=None,
        test_prefix=[
            '/sdb/data1/speech/24kHz/LibriTTS/',
            '/sdb/data1/speech/24kHz/LibriTTS/',
        ],
        test_max_samples=None,
        stage="test",
    )
    eval_result = evaluation.evaluation()
    print(eval_result)
