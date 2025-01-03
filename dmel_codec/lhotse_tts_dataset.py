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

        audio, audio_lens = collate_audio(cuts) # audio: (channel, audio_length)
        text = [cut.supervisions[0].text for cut in cuts]
        # audio_file = [cut.recording.id for cut in cuts]

        # return {"text": text, "audios": audio, "audio_lengths": audio_lens, "audio_file": audio_file}
        return {"text": text, "audios": audio, "audio_lengths": audio_lens}


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        train_recordings_paths: Sequence[str] | None = None,
        train_supervisions_paths: Sequence[str] | None = None,

        val_recordings_paths: Sequence[str] | None = None,
        val_supervisions_paths: Sequence[str] | None = None,

        test_recordings_paths: Sequence[str] | None = None,
        test_supervisions_paths: Sequence[str] | None = None,

        train_cuts_paths: Sequence[str] | None = None,
        val_cuts_paths: Sequence[str] | None = None,
        test_cuts_paths: Sequence[str] | None = None,

        # prefix for recording and supervision, Optional
        train_prefix: list[str] | None = None,
        val_prefix: list[str] | None = None,
        test_prefix: list[str] | None = None,

        train_max_duration: float = 60.0,  # dynamic batch size, seconds
        train_num_workers: int = 0,

        val_max_duration: float = 60.0,  # dynamic batch size, seconds
        val_num_workers: int = 0,
        val_max_samples: int = 128,

        # training stage just active train_dataloader and val_dataloader, None for not active
        test_max_duration: float | None = None,  # dynamic batch size, seconds
        test_num_workers: int | None = None,
        test_max_samples: int | None = None,
        world_size: int = 1,
        rank: int = 0,
        stage: str = "fit", # for hparams check
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.hparams_check()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def hparams_check(self):
        if self.hparams.stage == "fit":
            self._train_stage_hparams_check()
            self._val_stage_hparams_check()

        if self.hparams.stage == "validate":
            self._val_stage_hparams_check()
        
        if self.hparams.stage == "test":
            self._test_stage_hparams_check()

    def _train_stage_hparams_check(self):
        # train_recordings_paths and train_supervisions_paths and train_cuts_paths
        if self.hparams.train_recordings_paths is not None:
            assert len(self.hparams.train_recordings_paths) == len(self.hparams.train_supervisions_paths), "train_recordings_paths and train_supervisions_paths must have the same length"

        if self.hparams.train_recordings_paths is None:
            assert self.hparams.train_cuts_paths is not None and self.hparams.train_supervisions_paths is None, "train_cuts_paths must be provided and train_supervisions_paths must be None if train_recordings_paths is None"

        if self.hparams.train_cuts_paths is None:
            assert self.hparams.train_recordings_paths is not None, "train_recordings_paths must be provided if train_cuts_paths is None"

        # train_recordings_paths and train_cuts_paths and train_recording_prefix
        if self.hparams.train_recordings_paths is not None and self.hparams.train_cuts_paths is None:
            assert len(self.hparams.train_recordings_paths) == len(self.hparams.train_prefix), "train_recordings_paths and train_prefix must have the same length if train_cuts_paths is None"
        elif self.hparams.train_cuts_paths is not None and self.hparams.train_recordings_paths is not None:
            assert (len(self.hparams.train_cuts_paths) + len(self.hparams.train_recordings_paths)) == len(self.hparams.train_prefix), "train_cuts_paths + train_recordings_paths == train_prefix"

        # train_prefix == None
        if self.hparams.train_prefix is None:
            log.info(f"train_prefix is None, pls check your train_recordings source is absolute path")

    def _val_stage_hparams_check(self):
        # val_recordings_paths and val_supervisions_paths
        if self.hparams.val_recordings_paths is not None:
            assert len(self.hparams.val_recordings_paths) == len(self.hparams.val_supervisions_paths), "val_recordings_paths and val_supervisions_paths must have the same length"

        if self.hparams.val_recordings_paths is None:
            assert self.hparams.val_cuts_paths is not None and self.hparams.val_supervisions_paths is None, "val_cuts_paths must be provided and val_supervisions_paths must be None if val_recordings_paths is None"

        if self.hparams.val_cuts_paths is None:
            assert self.hparams.val_recordings_paths is not None, "val_recordings_paths must be provided if val_cuts_paths is None"

        # val_recordings_paths and val_cuts_paths and val_recording_prefix
        if self.hparams.val_recordings_paths is not None and self.hparams.val_cuts_paths is None:
            assert len(self.hparams.val_recordings_paths) == len(self.hparams.val_prefix), "val_recordings_paths and val_prefix must have the same length if val_cuts_paths is None"
        elif self.hparams.val_cuts_paths is not None and self.hparams.val_recordings_paths is not None:
            assert (len(self.hparams.val_cuts_paths) + len(self.hparams.val_recordings_paths)) == len(self.hparams.val_prefix), "val_cuts_paths + val_recordings_paths == val_prefix"

        # val_prefix == None
        if self.hparams.val_prefix is None:
            log.info(f"val_prefix is None, pls check your val_recordings source is absolute path")

        if self.hparams.val_max_samples is not None:
            log.info(f"validation stage just use {self.hparams.val_max_samples} samples")
        elif self.hparams.val_max_samples is None:
            log.info(f"validation stage use all samples")

    def _test_stage_hparams_check(self):
        # test_recordings_paths and test_supervisions_paths
        if self.hparams.test_recordings_paths is not None:
            assert len(self.hparams.test_recordings_paths) == len(self.hparams.test_supervisions_paths), "test_recordings_paths and test_supervisions_paths must have the same length"

        if self.hparams.test_recordings_paths is None:
            assert self.hparams.test_supervisions_paths is None, "test_supervisions_paths must be None if test_recordings_paths is None"

        # test_recordings_paths and test_cuts_paths and test_recording_prefix
        if self.hparams.test_recordings_paths is not None and self.hparams.test_cuts_paths is None:
            assert len(self.hparams.test_recordings_paths) == len(self.hparams.test_prefix), "test_recordings_paths and test_recording_prefix must have the same length if test_cuts_paths is None"
        elif self.hparams.test_cuts_paths is not None and self.hparams.test_recordings_paths is not None:
            assert (len(self.hparams.test_cuts_paths) + len(self.hparams.test_recordings_paths)) == len(self.hparams.test_prefix), "test_cuts_paths + test_recordings_paths == test_prefix"

        # test_prefix == None
        if self.hparams.test_prefix is None and (self.hparams.test_recordings_paths is not None or self.hparams.test_cuts_paths is not None):
            log.info(f"test_prefix is None, pls check your test_recordings source is absolute path")

        if self.hparams.test_max_samples is not None:
            log.info(f"test stage just use {self.hparams.test_max_samples} samples")
        elif self.hparams.test_max_samples is None:
            log.info(f"test stage use all samples")

    def setup(self, stage: str):
        # fit == train, need train and val dataset
        if stage == "fit":
            self._set_up_train_dataset()
            self._set_up_val_dataset()
        
        elif stage == "validate":
            self._set_up_val_dataset()
            
        elif stage == "test":
            self._set_up_test_dataset()

    def _set_up_train_dataset(self):
        train_recordings_length = len(self.hparams.train_recordings_paths) if self.hparams.train_recordings_paths is not None else 0
        train_cuts_length = len(self.hparams.train_cuts_paths) if self.hparams.train_cuts_paths is not None else 0
        self.train_cuts = CutSet()
        if train_recordings_length != 0:
            self.train_recordings = RecordingSet()
            self.train_supervisions = SupervisionSet()
            for idx, path in enumerate(self.hparams.train_recordings_paths):
                self.tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                if self.hparams.train_prefix is not None:
                    self.train_recordings += self.tmp_recordings.with_path_prefix(
                        self.hparams.train_prefix[idx]
                    )
                else:
                    self.train_recordings += self.tmp_recordings

            for path in self.hparams.train_supervisions_paths:
                self.train_supervisions += SupervisionSet.from_jsonl_lazy(path)

            log.info(f"train_recordings: {self.train_recordings}")
            log.info(f"train_supervisions: {self.train_supervisions}")

            self.train_cuts += CutSet.from_manifests(
                recordings=self.train_recordings, supervisions=self.train_supervisions
            )

        if train_cuts_length != 0:
            self.train_cuts_tmp = CutSet()
            for idx, path in enumerate(self.hparams.train_cuts_paths):
                self.tmp_cuts = CutSet.from_jsonl_lazy(path)
                self.train_cuts_tmp += self.tmp_cuts.with_recording_path_prefix(
                    self.hparams.train_prefix[idx + train_recordings_length]
                )
            self.train_cuts += self.train_cuts_tmp
        
        log.info(f"train_cuts: {self.train_cuts}")

        self.train_dataset = LhotseTTSDataset()
        log.info(f"train_dataset: {self.train_dataset}")

        self.train_sampler = DynamicBucketingSampler(
            self.train_cuts,
            max_duration=self.hparams.train_max_duration,
            shuffle=True,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"train_sampler: {self.train_sampler}")

    def _set_up_val_dataset(self):
        val_recordings_length = len(self.hparams.val_recordings_paths) if self.hparams.val_recordings_paths is not None else 0
        val_cuts_length = len(self.hparams.val_cuts_paths) if self.hparams.val_cuts_paths is not None else 0
        self.val_cuts = CutSet()
        if val_recordings_length != 0:
            self.val_recordings = RecordingSet()
            self.val_supervisions = SupervisionSet()
            for idx, path in enumerate(self.hparams.val_recordings_paths):
                self.tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                
                self.val_recordings += self.tmp_recordings.with_path_prefix(
                    self.hparams.val_prefix[idx]
                )

            for path in self.hparams.val_supervisions_paths:
                self.val_supervisions += SupervisionSet.from_jsonl_lazy(path)

            log.info(f"val_recordings: {self.val_recordings}")
            log.info(f"val_supervisions: {self.val_supervisions}")

            self.val_cuts += CutSet.from_manifests(
                recordings=self.val_recordings, supervisions=self.val_supervisions
            )

        if val_cuts_length != 0:
            self.val_cuts_tmp = CutSet()
            for idx, path in enumerate(self.hparams.val_cuts_paths):
                self.tmp_cuts = CutSet.from_jsonl_lazy(path)
                if self.hparams.val_prefix is not None:
                    self.val_cuts_tmp += self.tmp_cuts.with_recording_path_prefix(
                        self.hparams.val_prefix[idx + val_recordings_length]
                    )
                else:
                    self.val_cuts_tmp += self.tmp_cuts
            self.val_cuts += self.val_cuts_tmp

        self.val_dataset = LhotseTTSDataset()
        log.info(f"val_dataset: {self.val_dataset}")
        if self.hparams.val_max_samples is not None and len(self.val_cuts) > self.hparams.val_max_samples:
            self.val_cuts = CutSet.from_cuts(list(self.val_cuts)[: self.hparams.val_max_samples])

        self.val_sampler = DynamicBucketingSampler(
            self.val_cuts,
            max_duration=self.hparams.val_max_duration,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"val_sampler: {self.val_sampler}")

    def _set_up_test_dataset(self):
        test_recordings_length = len(self.hparams.test_recordings_paths) if self.hparams.test_recordings_paths is not None else 0
        test_cuts_length = len(self.hparams.test_cuts_paths) if self.hparams.test_cuts_paths is not None else 0
        self.test_cuts = CutSet()
        if test_recordings_length != 0:
            self.test_recordings = RecordingSet()
            self.test_supervisions = SupervisionSet()
            for idx, path in enumerate(self.hparams.test_recordings_paths):
                self.tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                if self.hparams.test_prefix is not None:
                    self.test_recordings += self.tmp_recordings.with_path_prefix(
                        self.hparams.test_prefix[idx]
                    )
                else:
                    self.test_recordings += self.tmp_recordings

            for path in self.hparams.test_supervisions_paths:
                self.test_supervisions += SupervisionSet.from_jsonl_lazy(path)

            log.info(f"test_recordings: {self.test_recordings}")
            log.info(f"test_supervisions: {self.test_supervisions}")

            self.test_cuts += CutSet.from_manifests(
                recordings=self.test_recordings, supervisions=self.test_supervisions
            )

        if test_cuts_length != 0:
            self.test_cuts_tmp = CutSet()
            for idx, path in enumerate(self.hparams.test_cuts_paths):
                self.tmp_cuts = CutSet.from_jsonl_lazy(path)
                if self.hparams.test_prefix is not None:
                    self.test_cuts_tmp += self.tmp_cuts.with_recording_path_prefix(
                        self.hparams.test_prefix[idx + test_recordings_length]
                    )
                else:
                    self.test_cuts_tmp += self.tmp_cuts
            self.test_cuts += self.test_cuts_tmp

        self.test_dataset = LhotseTTSDataset()
        log.info(f"test_dataset: {self.test_dataset}")

        if self.hparams.test_max_samples is not None and len(self.test_cuts) > self.hparams.test_max_samples:
            self.test_cuts = CutSet.from_cuts(list(self.test_cuts)[: self.hparams.test_max_samples])

        self.test_sampler = DynamicBucketingSampler(
            self.test_cuts,
            max_duration=self.hparams.test_max_duration,
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
        train_recordings_paths=[
            "/sdb/data1/lhotse/libritts/libritts_recordings_train-clean-100.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_recordings_train-clean-360.jsonl.gz",
        ],
        val_recordings_paths=[
            "/sdb/data1/lhotse/libritts/libritts_recordings_dev-clean.jsonl.gz"
        ],
        test_recordings_paths=[
            "/sdb/data1/lhotse/libritts/libritts_recordings_test-clean.jsonl.gz"
        ],
        train_supervisions_paths=[
            "/sdb/data1/lhotse/libritts/libritts_supervisions_train-clean-100.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_supervisions_train-clean-360.jsonl.gz",
        ],
        val_supervisions_paths=[
            "/sdb/data1/lhotse/libritts/libritts_supervisions_dev-clean.jsonl.gz"
        ],
        test_supervisions_paths=[
            "/sdb/data1/lhotse/libritts/libritts_supervisions_test-clean.jsonl.gz"
        ],
        train_cuts_paths=['/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz', '/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz'],
        val_cuts_paths=['/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz', '/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz'],
        test_cuts_paths=['/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz', '/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz'],

        train_prefix=[ # First specify the prefix for recordings, then specify the prefix for cuts
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/24kHz/Emilia",
            "/sdb/data1/speech/24kHz/Emilia",
        ],
        val_prefix=[ # First specify the prefix for recordings, then specify the prefix for cuts
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/24kHz/Emilia",
            "/sdb/data1/speech/24kHz/Emilia",
        ],
        test_prefix=[ # First specify the prefix for recordings, then specify the prefix for cuts
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/24kHz/Emilia",
            "/sdb/data1/speech/24kHz/Emilia",
        ],

        train_max_duration=60.0,
        train_num_workers=4,
        val_max_duration=60.0,
        val_num_workers=4,
        test_max_duration=60.0,
        test_num_workers=4,
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
