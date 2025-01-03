import logging
import sys
import os
from torch.utils.data import DataLoader, Dataset
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse import RecordingSet, SupervisionSet
from lhotse.dataset.collation import collate_audio
from lhotse.serialization import load_jsonl
from lightning import LightningDataModule
from dmel_codec.utils.utils import open_filelist
from dmel_codec.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class LhotseTTSDataset(Dataset):

    def __getitem__(self, cuts: CutSet):
        cuts = cuts.sort_by_duration(ascending=False)

        audio, audio_lens = collate_audio(cuts)  # audio: (channel, audio_length)
        text = [cut.supervisions[0].text for cut in cuts]
        # audio_file = [cut.recording.id for cut in cuts]

        # return {"text": text, "audios": audio, "audio_lengths": audio_lens, "audio_file": audio_file}
        return {"text": text, "audios": audio, "audio_lengths": audio_lens}


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        train_recordings_paths: list[str] | None = None,
        train_supervisions_paths: list[str] | None = None,
        train_recordings_filelist: list[str] | None = None,
        train_supervisions_filelist: list[str] | None = None,
        val_recordings_paths: list[str] | None = None,
        val_supervisions_paths: list[str] | None = None,
        val_recordings_filelist: list[str] | None = None,
        val_supervisions_filelist: list[str] | None = None,
        test_recordings_paths: list[str] | None = None,
        test_supervisions_paths: list[str] | None = None,
        test_recordings_filelist: list[str] | None = None,
        test_supervisions_filelist: list[str] | None = None,
        train_cuts_paths: list[str] | None = None,
        val_cuts_paths: list[str] | None = None,
        test_cuts_paths: list[str] | None = None,
        train_cuts_filelist: list[str] | None = None,
        val_cuts_filelist: list[str] | None = None,
        test_cuts_filelist: list[str] | None = None,
        # prefix for recording, Optional
        train_recordings_prefix: list[str] | None = None,
        val_recordings_prefix: list[str] | None = None,
        test_recordings_prefix: list[str] | None = None,
        # prefix for recording filelist, Optional
        train_recordings_filelist_prefix: list[str] | None = None,
        val_recordings_filelist_prefix: list[str] | None = None,
        test_recordings_filelist_prefix: list[str] | None = None,
        # prefix for cuts, Optional
        train_cuts_prefix: list[str] | None = None,
        val_cuts_prefix: list[str] | None = None,
        test_cuts_prefix: list[str] | None = None,
        # prefix for cuts filelist, Optional
        train_cuts_filelist_prefix: list[str] | None = None,
        val_cuts_filelist_prefix: list[str] | None = None,
        test_cuts_filelist_prefix: list[str] | None = None,
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
        stage: str = "fit",  # for hparams check
        output_dir: str | None = None,
        sample_rate: int = 24000,
    ):
        """
        recordings, supervisions, cuts, filelist:
            note: These four parameters are mutually exclusive, you can only provide one of them
            note: filelist means you provide a txt file, and the txt file contains the absolute path info

        prefix:
            note: all path in one filelist use the same prefix
            note: prefix can be None, if you are absolute path in your audio source

        stage: fit, validate, test
            note: if you are fit stage, you must provide train and val info
            note: if you are validate stage, you must provide val info
            note: if you are test stage, you must provide test info

        max_duration: float = 60.0
            note: dynamic batch size, seconds
        max_samples: int = 128
            note: for fit stage, only use max_samples samples to evaluate, speed up train

        world_size: int = 1
            note: for distributed training

        output_dir: str | None = None
            note: Optional, for save cutset, for resume training,
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.hparams_check()  # check hparams and load filelist if filelist is not None

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

    def setup(self, stage: str):
        # fit == train, need train and val dataset
        if stage == "fit":
            self._set_up_train_dataset()
            self._set_up_val_dataset()

        elif stage == "validate":
            self._set_up_val_dataset()

        elif stage == "test":
            self._set_up_test_dataset()

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

    # check train hparams and load train filelist if train filelist is not None
    def _train_stage_hparams_check(self):
        # train_recordings_filelist and train_supervisions_filelist and train_cuts_filelist
        if self.hparams.train_recordings_filelist is not None:
            assert len(self.hparams.train_recordings_filelist) == len(
                self.hparams.train_supervisions_filelist
            ), "train_recordings_filelist and train_supervisions_filelist must have the same length"
            if self.hparams.train_recordings_prefix is not None:
                assert len(self.hparams.train_recordings_filelist) == len(
                    self.hparams.train_recordings_prefix
                ), "train_recordings_filelist and train_recordings_prefix must have the same length"
            self.hparams.train_recordings_paths_list = []
            self.hparams.train_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.train_recordings_filelist,
                self.hparams.train_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.train_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.train_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.train_recordings_paths_list = None
            self.hparams.train_supervisions_paths_list = None

        if self.hparams.train_cuts_filelist is not None:
            self.hparams.train_cuts_paths_list = []
            for path in self.hparams.train_cuts_filelist:
                self.hparams.train_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.train_cuts_paths_list = None

        # train_recordings_paths and train_supervisions_paths and train_cuts_paths
        if self.hparams.train_recordings_paths is not None:
            assert len(self.hparams.train_recordings_paths) == len(
                self.hparams.train_supervisions_paths
            ), "train_recordings_paths and train_supervisions_paths must have the same length"
            if self.hparams.train_recordings_prefix is not None:
                assert len(self.hparams.train_recordings_paths) == len(
                    self.hparams.train_recordings_prefix
                ), "train_recordings_paths and train_recordings_prefix must have the same length"

        # train_prefix == None
        if self.hparams.train_recordings_prefix is None and self.hparams.train_recordings_paths is not None:
            log.info(
                f"train_recordings_prefix is None, pls check your train_recordings source is absolute path"
            )

        if self.hparams.train_cuts_prefix is None and self.hparams.train_cuts_paths is not None:
            log.info(
                f"train_cuts_prefix is None, pls check your train_cuts source is absolute path"
            )

        if self.hparams.train_recordings_filelist_prefix is None and self.hparams.train_recordings_filelist is not None:
            log.info(
                f"train_recordings_filelist_prefix is None, pls check your train_recordings_filelist source is absolute path"
            )

        if self.hparams.train_cuts_filelist_prefix is None and self.hparams.train_cuts_filelist is not None:
            log.info(
                f"train_cuts_filelist_prefix is None, pls check your train_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.train_recordings_paths is None
            and self.hparams.train_cuts_paths is None
            and self.hparams.train_recordings_filelist is None
            and self.hparams.train_cuts_filelist is None
        ):
            raise ValueError(
                "train_recordings_paths, train_cuts_paths, train_recordings_filelist, train_cuts_filelist must be provided at least one"
            )


    # check val hparams and load val filelist if val filelist is not None
    def _val_stage_hparams_check(self):
        # val_recordings_filelist and val_supervisions_filelist and val_cuts_filelist
        if self.hparams.val_recordings_filelist is not None:
            assert len(self.hparams.val_recordings_filelist) == len(
                self.hparams.val_supervisions_filelist
            ), "val_recordings_filelist and val_supervisions_filelist must have the same length"
            if self.hparams.val_recordings_prefix is not None:
                assert len(self.hparams.val_recordings_filelist) == len(
                    self.hparams.val_recordings_prefix
                ), "val_recordings_filelist and val_recordings_prefix must have the same length"
            self.hparams.val_recordings_paths_list = []
            self.hparams.val_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.val_recordings_filelist,
                self.hparams.val_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.val_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.val_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.val_recordings_paths_list = None
            self.hparams.val_supervisions_paths_list = None

        if self.hparams.val_cuts_filelist is not None:
            self.hparams.val_cuts_paths_list = []
            for path in self.hparams.val_cuts_filelist:
                self.hparams.val_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.val_cuts_paths_list = None

        # val_recordings_paths and val_supervisions_paths and val_cuts_paths
        if self.hparams.val_recordings_paths is not None:
            assert len(self.hparams.val_recordings_paths) == len(
                self.hparams.val_supervisions_paths
            ), "val_recordings_paths and val_supervisions_paths must have the same length"
            if self.hparams.val_recordings_prefix is not None:
                assert len(self.hparams.val_recordings_paths) == len(
                    self.hparams.val_recordings_prefix
                ), "val_recordings_paths and val_recordings_prefix must have the same length"

        # val_prefix == None
        if self.hparams.val_recordings_prefix is None and self.hparams.val_recordings_paths is not None:
            log.info(
                f"val_recordings_prefix is None, pls check your val_recordings source is absolute path"
            )

        if self.hparams.val_cuts_prefix is None and self.hparams.val_cuts_paths is not None:
            log.info(
                f"val_cuts_prefix is None, pls check your val_cuts source is absolute path"
            )

        if self.hparams.val_recordings_filelist_prefix is None and self.hparams.val_recordings_filelist is not None:
            log.info(
                f"val_recordings_filelist_prefix is None, pls check your val_recordings_filelist source is absolute path"
            )

        if self.hparams.val_cuts_filelist_prefix is None and self.hparams.val_cuts_filelist is not None:
            log.info(
                f"val_cuts_filelist_prefix is None, pls check your val_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.val_recordings_paths is None
            and self.hparams.val_cuts_paths is None
            and self.hparams.val_recordings_filelist is None
            and self.hparams.val_cuts_filelist is None
        ):
            raise ValueError(
                "val_recordings_paths, val_cuts_paths, val_recordings_filelist, val_cuts_filelist must be provided at least one"
            )

        if self.hparams.val_max_samples is not None:
            log.info(f"val stage just use {self.hparams.val_max_samples} samples")
        elif self.hparams.val_max_samples is None:
            log.info(f"val stage use all samples")

    # check test hparams and load test filelist if test filelist is not None
    def _test_stage_hparams_check(self):
        # test_recordings_filelist and test_supervisions_filelist and test_cuts_filelist
        if self.hparams.test_recordings_filelist is not None:
            assert len(self.hparams.test_recordings_filelist) == len(
                self.hparams.test_supervisions_filelist
            ), "test_recordings_filelist and test_supervisions_filelist must have the same length"
            if self.hparams.test_recordings_prefix is not None:
                assert len(self.hparams.test_recordings_filelist) == len(
                    self.hparams.test_recordings_prefix
                ), "test_recordings_filelist and test_recordings_prefix must have the same length"
            self.hparams.test_recordings_paths_list = []
            self.hparams.test_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.test_recordings_filelist,
                self.hparams.test_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.test_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.test_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.test_recordings_paths_list = None
            self.hparams.test_supervisions_paths_list = None

        if self.hparams.test_cuts_filelist is not None:
            self.hparams.test_cuts_paths_list = []
            for path in self.hparams.test_cuts_filelist:
                self.hparams.test_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.test_cuts_paths_list = None

        # train_recordings_paths and train_supervisions_paths and train_cuts_paths
        if self.hparams.test_recordings_paths is not None:
            assert len(self.hparams.test_recordings_paths) == len(
                self.hparams.test_supervisions_paths
            ), "test_recordings_paths and test_supervisions_paths must have the same length"
            if self.hparams.test_recordings_prefix is not None:
                assert len(self.hparams.test_recordings_paths) == len(
                    self.hparams.test_recordings_prefix
                ), "test_recordings_paths and test_recordings_prefix must have the same length"

        # train_prefix == None
        if self.hparams.test_recordings_prefix is None and self.hparams.test_recordings_paths is not None:
            log.info(
                f"test_recordings_prefix is None, pls check your test_recordings source is absolute path"
            )

        if self.hparams.test_cuts_prefix is None and self.hparams.test_cuts_paths is not None:
            log.info(
                f"test_cuts_prefix is None, pls check your test_cuts source is absolute path"
            )

        if self.hparams.test_recordings_filelist_prefix is None and self.hparams.test_recordings_filelist is not None:
            log.info(
                f"test_recordings_filelist_prefix is None, pls check your test_recordings_filelist source is absolute path"
            )

        if self.hparams.test_cuts_filelist_prefix is None and self.hparams.test_cuts_filelist is not None:
            log.info(
                f"test_cuts_filelist_prefix is None, pls check your test_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.test_recordings_paths is None
            and self.hparams.test_cuts_paths is None
            and self.hparams.test_recordings_filelist is None
            and self.hparams.test_cuts_filelist is None
        ):
            raise ValueError(
                "test_recordings_paths, test_cuts_paths, test_recordings_filelist, test_cuts_filelist must be provided at least one"
            )

        if self.hparams.test_max_samples is not None:
            log.info(f"test stage just use {self.hparams.test_max_samples} samples")
        elif self.hparams.test_max_samples is None:
            log.info(f"test stage use all samples")

    # load train dataset
    def _set_up_train_dataset(self):
        # train_recordings and train_supervisions
        self.train_cuts = CutSet()
        self.train_recordings = RecordingSet()
        self.train_supervisions = SupervisionSet()
        if self.hparams.train_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.train_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.train_recordings_prefix is not None:
                    prefix = self.hparams.train_recordings_prefix[idx]
                    if prefix != '':
                        self.train_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.train_recordings += tmp_recordings
                else:
                    self.train_recordings += tmp_recordings

            for path in self.hparams.train_supervisions_paths:
                self.train_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # train_recordings_filelist and train_supervisions_filelist
        if self.hparams.train_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.train_recordings_paths_list
            ):  # self.hparams.train_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.train_recordings_filelist_prefix is not None:
                    prefix = self.hparams.train_recordings_filelist_prefix[idx]
                    if prefix != '':
                        self.train_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.train_recordings += tmp_recordings
                else:
                    self.train_recordings += tmp_recordings

            # self.hparams.train_supervisions_paths_list: [[xxx], [xxx]]
            for path_list in self.hparams.train_supervisions_paths_list:  
                for path in path_list:
                    self.train_supervisions += SupervisionSet.from_jsonl_lazy(path)

        if (self.hparams.train_recordings_paths is not None) or (self.hparams.train_recordings_filelist is not None):
            log.info(f"train_recordings: {self.train_recordings}")
            log.info(f"train_supervisions: {self.train_supervisions}")

            self.train_cuts += CutSet.from_manifests(
                recordings=self.train_recordings, supervisions=self.train_supervisions
            )

        # train_cuts
        if self.hparams.train_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.train_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.train_cuts_prefix is not None:
                    prefix = self.hparams.train_cuts_prefix[idx]
                    if prefix != '':
                        self.train_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.train_cuts += tmp_cuts
                else:
                    self.train_cuts += tmp_cuts

        # train_cuts_filelist
        if self.hparams.train_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.train_cuts_paths_list
            ):  # self.hparams.train_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                for path in path_list:
                    tmp_cuts += CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.train_cuts_filelist_prefix is not None:
                    prefix = self.hparams.train_cuts_filelist_prefix[idx]
                    if prefix != '':
                        self.train_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.train_cuts += tmp_cuts
                else:
                    self.train_cuts += tmp_cuts

        log.info(f"train_cuts: {self.train_cuts}")

        if self.hparams.output_dir is not None:
            self.train_cuts.to_file(
                os.path.join(self.hparams.output_dir, "train_cuts.jsonl.gz")
            )

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

    # load val dataset
    def _set_up_val_dataset(self):
        self.val_cuts = CutSet()
        self.val_recordings = RecordingSet()
        self.val_supervisions = SupervisionSet()
        # val_recordings and val_supervisions
        if self.hparams.val_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.val_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.val_recordings_prefix is not None:
                    prefix = self.hparams.val_recordings_prefix[idx]
                    if prefix != '':
                        self.val_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.val_recordings += tmp_recordings
                else:
                    self.val_recordings += tmp_recordings

            for path in self.hparams.val_supervisions_paths:
                self.val_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # val_recordings_filelist and val_supervisions_filelist
        if self.hparams.val_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.val_recordings_paths_list
            ):  # self.hparams.val_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.val_recordings_filelist_prefix is not None:
                    prefix = self.hparams.val_recordings_filelist_prefix[idx]
                    if prefix != '':
                        self.val_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.val_recordings += tmp_recordings
                else:
                    self.val_recordings += tmp_recordings

        if (self.hparams.val_recordings_paths is not None) or (self.hparams.val_recordings_filelist is not None):
            log.info(f"val_recordings: {self.val_recordings}")
            log.info(f"val_supervisions: {self.val_supervisions}")

            self.val_cuts += CutSet.from_manifests(
                recordings=self.val_recordings, supervisions=self.val_supervisions
            )

        # val_cuts
        if self.hparams.val_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.val_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.val_cuts_prefix is not None:
                    prefix = self.hparams.val_cuts_prefix[idx]
                    if prefix != '':
                        self.val_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.val_cuts += tmp_cuts
                else:
                    self.val_cuts += tmp_cuts

        # val_cuts_filelist
        if self.hparams.val_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.val_cuts_paths_list
            ):  # self.hparams.val_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                for path in path_list:
                    tmp_cuts += CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.val_cuts_filelist_prefix is not None:
                    prefix = self.hparams.val_cuts_filelist_prefix[idx]
                    if prefix != '':
                        self.val_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.val_cuts += tmp_cuts
                else:
                    self.val_cuts += tmp_cuts

        if self.hparams.output_dir is not None:
            self.val_cuts.to_file(
                os.path.join(self.hparams.output_dir, "val_cuts.jsonl.gz")
            )

        self.val_dataset = LhotseTTSDataset()
        log.info(f"val_dataset: {self.val_dataset}")
        if (
            self.hparams.val_max_samples is not None
            and len(self.val_cuts) > self.hparams.val_max_samples
        ):
            self.val_cuts = CutSet.from_cuts(
                list(self.val_cuts)[: self.hparams.val_max_samples]
            )

        self.val_sampler = DynamicBucketingSampler(
            self.val_cuts,
            max_duration=self.hparams.val_max_duration,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"val_sampler: {self.val_sampler}")

    # load test dataset
    def _set_up_test_dataset(self):
        self.test_cuts = CutSet()
        self.test_recordings = RecordingSet()
        self.test_supervisions = SupervisionSet()
        # test_recordings and test_supervisions
        if self.hparams.test_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.test_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.test_recordings_prefix is not None:
                    prefix = self.hparams.test_recordings_prefix[idx]
                    if prefix != '':
                        self.test_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.test_recordings += tmp_recordings
                else:
                    self.test_recordings += tmp_recordings

            for path in self.hparams.test_supervisions_paths:
                self.test_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # test_recordings_filelist
        if self.hparams.test_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.test_recordings_paths_list
            ):  # self.hparams.test_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.test_recordings_filelist_prefix is not None:
                    prefix = self.hparams.test_recordings_filelist_prefix[idx]
                    if prefix != '':
                        self.test_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.test_recordings += tmp_recordings
                else:
                    self.test_recordings += tmp_recordings

        if (self.hparams.test_recordings_paths is not None) or (self.hparams.test_recordings_filelist is not None):
            log.info(f"test_recordings: {self.test_recordings}")
            log.info(f"test_supervisions: {self.test_supervisions}")

            self.test_cuts += CutSet.from_manifests(
                recordings=self.test_recordings, supervisions=self.test_supervisions
            )

        # test_cuts
        if self.hparams.test_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.test_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.test_cuts_prefix is not None:
                    prefix = self.hparams.test_cuts_prefix[idx]
                    if prefix != '':
                        self.test_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.test_cuts += tmp_cuts

        # test_cuts_filelist
        if self.hparams.test_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.test_cuts_paths_list
            ):  # self.hparams.test_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                for path in path_list:
                    tmp_cuts += CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.test_cuts_filelist_prefix is not None:
                    prefix = self.hparams.test_cuts_filelist_prefix[idx]
                    if prefix != '':
                        self.test_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.test_cuts += tmp_cuts

        self.test_dataset = LhotseTTSDataset()
        log.info(f"test_dataset: {self.test_dataset}")

        if self.hparams.output_dir is not None:
            self.test_cuts.to_file(
                os.path.join(self.hparams.output_dir, "test_cuts.jsonl.gz")
            )

        if (
            self.hparams.test_max_samples is not None
            and len(self.test_cuts) > self.hparams.test_max_samples
        ):
            self.test_cuts = CutSet.from_cuts(
                list(self.test_cuts)[: self.hparams.test_max_samples]
            )

        self.test_sampler = DynamicBucketingSampler(
            self.test_cuts,
            max_duration=self.hparams.test_max_duration,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"test_sampler: {self.test_sampler}")

        if self.hparams.output_dir is not None:
            self.test_cuts.to_file(
                os.path.join(self.hparams.output_dir, "test_cuts.jsonl.gz")
            )

if __name__ == "__main__":
    # test
    data_module = LhotseDataModule(
        train_recordings_paths=[
            "/sdb/data1/lhotse/libritts/libritts_recordings_train-clean-100.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_recordings_train-clean-360.jsonl.gz",
            "/sdb/data1/lhotse/aishell3/aishell3_recordings_train.jsonl.gz",
        ],
        train_supervisions_paths=[
            "/sdb/data1/lhotse/libritts/libritts_supervisions_train-clean-100.jsonl.gz",
            "/sdb/data1/lhotse/libritts/libritts_supervisions_train-clean-360.jsonl.gz",
            "/sdb/data1/lhotse/aishell3/aishell3_supervisions_train.jsonl.gz",
        ],
        train_recordings_prefix=[  
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/24kHz/LibriTTS",
            "/sdb/data1/speech/44.1kHz/Aishell3",
        ],
        
        train_cuts_paths=[
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz",
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz",
        ],
        train_cuts_prefix=[  
            "/sdb/data1/speech/24kHz/Emilia",
            "/sdb/data1/speech/24kHz/Emilia",
        ],

        train_cuts_filelist=["/sdb/data1/lhotse/filelist/emilia/EN/filelist.txt"],
        train_cuts_filelist_prefix=["/sdb/data1/speech/24kHz/Emilia"],

        val_recordings_paths=["/sdb/data1/lhotse/libritts/libritts_recordings_dev-clean.jsonl.gz"],
        val_supervisions_paths=["/sdb/data1/lhotse/libritts/libritts_supervisions_dev-clean.jsonl.gz"],
        val_recordings_prefix=["/sdb/data1/speech/24kHz/LibriTTS"],

        val_cuts_paths=[
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz",
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz",
        ],
        val_cuts_prefix=[  
            "/sdb/data1/speech/24kHz/Emilia",
            "/sdb/data1/speech/24kHz/Emilia",
        ],

        test_recordings_paths=["/sdb/data1/lhotse/libritts/libritts_recordings_test-clean.jsonl.gz"],
        test_supervisions_paths=["/sdb/data1/lhotse/libritts/libritts_supervisions_test-clean.jsonl.gz"],
        test_recordings_prefix=["/sdb/data1/speech/24kHz/LibriTTS"],

        test_cuts_paths=[
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00000.jsonl.gz",
            "/sdb/data1/lhotse/emilia-lhotse/EN/EN_cuts_B00019.jsonl.gz",
        ],
        test_cuts_prefix=[  
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
