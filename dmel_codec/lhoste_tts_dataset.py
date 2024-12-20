import logging
import sys
from typing import Sequence

from torch.utils.data import DataLoader, Dataset
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse import RecordingSet, SupervisionSet
from lhotse.dataset.collation import collate_audio
from lhotse.serialization import load_jsonl
from lightning import LightningDataModule

from dmel_codec.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class LhotseTTSDataset(Dataset):

    def __getitem__(self, cuts: CutSet):
        cuts = cuts.sort_by_duration(ascending=False)

        audio, audio_lens = collate_audio(cuts)

        text = [cut.supervisions[0].text for cut in cuts]

        return {"text": text, "wav": audio, "duration": audio_lens}


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        train_recordings_paths: Sequence[str],
        val_recordings_paths: Sequence[str],
        test_recordings_paths: Sequence[str],
        train_supervisions_paths: Sequence[str],
        val_supervisions_paths: Sequence[str],
        test_supervisions_paths: Sequence[str],
        output_dir: str = "./data/libriTTS",
        json_prefix: str = "",
        recording_prefix: str = None,
        max_duration: float = 60.0,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
        test_num_workers: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_recordings = RecordingSet()
            for path in self.hparams.train_recordings_paths:
                self.train_recordings += RecordingSet.from_dicts(load_jsonl(path))
            self.train_supervisions = SupervisionSet()
            for path in self.hparams.train_supervisions_paths:
                self.train_supervisions += SupervisionSet.from_dicts(load_jsonl(path))
            if self.hparams.recording_prefix is not None:
                self.train_recordings = self.train_recordings.with_path_prefix(self.hparams.recording_prefix)
            log.info(f"train_recordings: {self.train_recordings}")
            log.info(f"train_supervisions: {self.train_supervisions}")
            
            self.train_cuts = CutSet.from_manifests(recordings=self.train_recordings, supervisions=self.train_supervisions)
            log.info(f"train_cuts: {self.train_cuts}")
            
            self.train_dataset = LhotseTTSDataset()
            log.info(f"train_dataset: {self.train_dataset}")
            
            self.train_sampler = DynamicBucketingSampler(
                self.train_cuts,
                max_duration=self.hparams.max_duration,
                shuffle=True,
                drop_last=False,
            )
            log.info(f"train_sampler: {self.train_sampler}")
        elif stage == "validate":
            self.val_recordings = RecordingSet()
            for path in self.hparams.val_recordings_paths:
                self.val_recordings += RecordingSet.from_dicts(load_jsonl(path))
            self.val_supervisions = SupervisionSet()
            for path in self.hparams.val_supervisions_paths:
                self.val_supervisions += SupervisionSet.from_dicts(load_jsonl(path))
            if self.hparams.recording_prefix is not None:
                self.val_recordings = self.val_recordings.with_path_prefix(self.hparams.recording_prefix)
            log.info(f"val_recordings: {self.val_recordings}")
            log.info(f"val_supervisions: {self.val_supervisions}")
            
            self.val_cuts = CutSet.from_manifests(recordings=self.val_recordings, supervisions=self.val_supervisions)
            log.info(f"val_cuts: {self.val_cuts}")
            
            self.val_dataset = LhotseTTSDataset()
            log.info(f"val_dataset: {self.val_dataset}")
            
            self.val_sampler = DynamicBucketingSampler(
                self.val_cuts,
                max_duration=self.hparams.max_duration,
                shuffle=False,
                drop_last=False,
            )
            log.info(f"val_sampler: {self.val_sampler}")
        elif stage == "test":
            self.test_recordings = RecordingSet()
            for path in self.hparams.test_recordings_paths:
                self.test_recordings += RecordingSet.from_dicts(load_jsonl(path))
            self.test_supervisions = SupervisionSet()
            for path in self.hparams.test_supervisions_paths:
                self.test_supervisions += SupervisionSet.from_dicts(load_jsonl(path))
            if self.hparams.recording_prefix is not None:
                self.test_recordings = self.test_recordings.with_path_prefix(self.hparams.recording_prefix)
            log.info(f"test_recordings: {self.test_recordings}")
            log.info(f"test_supervisions: {self.test_supervisions}")
            
            self.test_cuts = CutSet.from_manifests(recordings=self.test_recordings, supervisions=self.test_supervisions)
            log.info(f"test_cuts: {self.test_cuts}")

            self.test_dataset = LhotseTTSDataset()
            log.info(f"test_dataset: {self.test_dataset}")
            
            self.test_sampler = DynamicBucketingSampler(
                self.test_cuts,
                max_duration=self.hparams.max_duration,
                shuffle=False,
                drop_last=False,
            )
            log.info(f"test_sampler: {self.test_sampler}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            num_workers=self.hparams.train_num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            num_workers=self.hparams.val_num_workers,
            pin_memory=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            num_workers=self.hparams.test_num_workers,
            pin_memory=False,
        )


if __name__ == "__main__":
    # test

    data_module = LhotseDataModule(
        train_recordings_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_recordings_train-clean-100.jsonl.gz",
                                "/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_recordings_train-clean-360.jsonl.gz"],
        val_recordings_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_recordings_dev-clean.jsonl.gz"],
        test_recordings_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_recordings_test-clean.jsonl.gz"],
        train_supervisions_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_supervisions_train-clean-100.jsonl.gz",
                                "/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_supervisions_train-clean-360.jsonl.gz"],
        val_supervisions_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_supervisions_dev-clean.jsonl.gz"],
        test_supervisions_paths=["/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS/libritts_supervisions_test-clean.jsonl.gz"],
        output_dir="/home/4T/zhouhao/dMelChatMusic/chat_music/data/libriTTS",
        json_prefix="libritts",
        recording_prefix="magia",
        max_duration=60.0,
        train_num_workers=4,
        val_num_workers=4,
    )

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    log.info(f"train_loader: {train_loader}")
    cnt = 0
    for batch in train_loader:
        log.info(f"cnt: {cnt}")
        log.info(batch.keys())
        log.info(f"test: {batch['text']}")
        log.info(f"wav: {batch['wav'].shape}")
        log.info(f"duration: {batch['duration'].shape}")
        cnt += 1
        if cnt > 10:
            break

    data_module.setup("validate")
    val_loader = data_module.val_dataloader()
    log.info(f"val_loader: {val_loader}")
    cnt = 0
    for batch in val_loader:
        log.info(f"cnt: {cnt}")
        log.info(batch.keys())
        log.info(f"test: {batch['text']}")
        log.info(f"wav: {batch['wav'].shape}")
        log.info(f"duration: {batch['duration'].shape}")
        cnt += 1
        if cnt > 10:
            break
    
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    log.info(f"test_loader: {test_loader}")
    cnt = 0
    for batch in test_loader:
        log.info(f"cnt: {cnt}")
        log.info(batch.keys())
        log.info(f"test: {batch['text']}")
        log.info(f"wav: {batch['wav'].shape}")
        log.info(f"duration: {batch['duration'].shape}")
        cnt += 1
        if cnt > 10:
            break
