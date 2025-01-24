import logging
import sys
from torch.utils.data import DataLoader, Dataset
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lightning import LightningDataModule
from dmel_codec.utils.logger import RankedLogger
from lhotse import CutSet
import librosa
import torch
import warnings

log = RankedLogger(__name__, rank_zero_only=False)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class LhotseTTSDataset(Dataset):

    def __getitem__(self, cuts: CutSet):
        cuts = cuts.sort_by_duration(ascending=False)

        # same with bigvgan
        audio_list = []
        audio_lens = []
        audio_paths = []
        for cut in cuts:
            # audio = np.array(audio_length, )
            audio_path = cut.recording.sources[0].source
            audio, _ = librosa.load(
                audio_path, sr=cut.sampling_rate, mono=True, offset=cut.start, duration=cut.duration
            )
            audio = librosa.util.normalize(audio) * 0.95
            audio = torch.FloatTensor(audio)
            audio_list.append(audio)
            audio_lens.append(audio.shape[0])
            audio_paths.append(audio_path)


        return {
            "audios": audio_list,
            "audio_lengths": torch.tensor(audio_lens, dtype=torch.int32),
            "audio_paths": audio_paths,
        }

    def collate_fn(self, batch):
        audio_list = batch[0]["audios"]
        audio_lens = batch[0]["audio_lengths"]
        max_length = max(audio_lens)

        # right pad audio
        audio_list = [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[-1]))
            for audio in audio_list
        ]
        audios = torch.stack(audio_list, dim=0)
        if audios.ndim == 2:
            audios = audios.unsqueeze(1)

        return {
            "text": batch[0]["text"],
            "audios": audios,
            "audio_lengths": audio_lens.reshape(1, -1),
            "audio_paths": batch[0]["audio_paths"],
        }


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        stage: str,
        train_cuts_path: str | None = None,
        val_cuts_path: str | None = None,
        test_cuts_path: str | None = None,

        # sampler, dataloader hparams
        train_max_durations: str | None = None,
        val_max_durations: str | None = None,
        test_max_durations: str | None = None,
        train_num_workers: str | None = None,
        val_num_workers: str | None = None,
        test_num_workers: str | None = None,
        pin_memory: bool = False,
        world_size: int | None = None,
    ):
        """
        stage: fit, validate, test, required=True
            note: if you are fit stage, you must provide train and val info
            note: if you are validate stage, you must provide val info
            note: if you are test stage, you must provide test info

        train_cuts_path: str | None = None
            note: train cutset path

        val_cuts_path: str | None = None
            note: val cutset path

        test_cuts_path: str | None = None
            note: test cutset path
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.hparams_check()  # check hparams and load filelist if filelist is not None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def hparams_check(self):
        assert self.hparams.stage in ['fit', 'validate', 'test'], "stage must in [fit, validate, test]"
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
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            num_workers=self.hparams.train_num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            sampler=self.val_sampler,
            num_workers=self.hparams.val_num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            sampler=self.test_sampler,
            num_workers=self.hparams.test_num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.test_dataset.collate_fn,
        )

    # check train hparams and load train filelist if train filelist is not None
    def _train_stage_hparams_check(self):
        assert self.hparams.train_cuts_path is not None, "train_cuts_path must be provided"
        assert self.hparams.train_max_durations is not None, "train_max_durations must be provided"
        assert self.hparams.train_num_workers is not None, "train_num_workers must be provided"

    # check val hparams and load val filelist if val filelist is not None
    def _val_stage_hparams_check(self):
        assert self.hparams.val_cuts_path is not None, "val_cuts_path must be provided"
        assert self.hparams.val_max_durations is not None, "val_max_durations must be provided"
        assert self.hparams.val_num_workers is not None, "val_num_workers must be provided"

    # check test hparams and load test filelist if test filelist is not None
    def _test_stage_hparams_check(self):
        assert self.hparams.test_cuts_path is not None, "test_cuts_path must be provided"
        assert self.hparams.test_max_durations is not None, "test_max_durations must be provided"
        assert self.hparams.test_num_workers is not None, "test_num_workers must be provided"

    # load train dataset
    def _set_up_train_dataset(self):
        train_cut = CutSet.from_jsonl_lazy(self.hparams.train_cuts_path)
        self.train_dataset = LhotseTTSDataset()
        self.train_sampler = DynamicBucketingSampler(
            train_cut,
            max_duration=self.hparams.train_max_durations,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
            buffer_size=50000,
        )
        log.info(f"train_sampler: {self.train_sampler}")

    # load val dataset
    def _set_up_val_dataset(self):
        val_cut = CutSet.from_jsonl_lazy(self.hparams.val_cuts_path)
        self.val_dataset = LhotseTTSDataset()
        self.val_sampler = DynamicBucketingSampler(
            val_cut,
            max_duration=self.hparams.val_max_durations,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"val_sampler: {self.val_sampler}")

    # load test dataset
    def _set_up_test_dataset(self):
        test_cut = CutSet.from_jsonl_lazy(self.hparams.test_cuts_path)
        self.test_dataset = LhotseTTSDataset()
        self.test_sampler = DynamicBucketingSampler(
            test_cut,
            max_duration=self.hparams.test_max_durations,
            shuffle=False,
            drop_last=False,
            world_size=self.hparams.world_size,
        )
        log.info(f"test_sampler: {self.test_sampler}")

if __name__ == "__main__":
    from tqdm import tqdm
    # test
    data_module = LhotseDataModule(
        stage = 'fit',
        train_cuts_path='/home/wzy/projects/dmel_codec/train_cuts_windows-3_min_duration-3.0_max_duration-None.jsonl.gz',
        val_cuts_path='/home/wzy/projects/dmel_codec/val_cuts_sample-128.jsonl.gz',
        train_max_durations=210,
        val_max_durations=5,
        train_num_workers=10,
        val_num_workers=1,
    )

    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    log.info(f"train_loader: {train_loader}")
    cnt = 0
    import time
    start_time = time.time()
    for batch in tqdm(train_loader):
        if cnt % 1000 == 0:
            end_time = time.time()
            log.info(f"data loading time: {end_time - start_time}")
            start_time = end_time
            log.info(f"train stage cnt: {cnt}")
            log.info(batch.keys())
            # log.info(f"train stage text: {batch['text'][0]}")
            log.info(f"train stage audios: {batch['audios'].shape}")
            # log.info(f"train stage audio_lengths: {batch['audio_lengths'].shape}")
            # log.info(f"train stage audio_lengths: {batch['audio_lengths']}")
            # log.info(f"train stage audio_paths: {batch['audio_paths'][-5:]}")

        cnt += 1

    data_module.setup('validate')
    val_loader = data_module.val_dataloader()
    log.info(f"val_loader: {val_loader}")
    cnt = 0
    for batch in val_loader:
        log.info(f"val stage cnt: {cnt}")
        log.info(batch.keys())
        # log.info(f"val stage text: {batch['text']}")
        log.info(f"val stage audios: {batch['audios'].shape}")
        log.info(f"val stage audio_lengths: {batch['audio_lengths'].shape}")
        # log.info(f"val stage audio_paths: {batch['audio_paths']}")
        cnt += 1
        if cnt > 10:
            break