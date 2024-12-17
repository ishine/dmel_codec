import logging
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler
from lhotse.dataset.collation import collate_audio
from lightning import LightningDataModule

from models.configuration_qwen2 import Qwen2Config
from utils.logger import RankedLogger
from utils.tokenization_qwen2 import Qwen2Tokenizer

log = RankedLogger(__name__, rank_zero_only=False)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class LhotseTTSDataset(Dataset):
    def __init__(
        self,
        text_tokenizer: Qwen2Tokenizer,
        model_config: Qwen2Config,
    ):
        self.model_config = model_config
        self.text_tokenizer = text_tokenizer

    def __getitem__(self, cuts: CutSet):
        cuts = cuts.sort_by_duration(ascending=False)

        audio, audio_lens = collate_audio(cuts)

        # 获取文本并tokenize
        text = [cut.supervisions[0].text for cut in cuts]
        with torch.no_grad():
            text_logits = self.text_tokenizer.__call__(
                text, return_tensors="pt", padding=True
            )["input_ids"]

        return {"text_logits": text_logits, "wav": audio, "duration": audio_lens}


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        train_cuts: CutSet,
        val_cuts: CutSet,
        text_tokenizer: Qwen2Tokenizer,
        model_config: Qwen2Config,
        max_duration: float = 60.0,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.text_tokenizer = self.hparams.text_tokenizer
        self.model_config = self.hparams.model_config

        self.train_dataset = None
        self.val_dataset = None

        self.train_num_workers = self.hparams.train_num_workers
        self.val_num_workers = self.hparams.val_num_workers

    def setup(self, stage: str):
        if stage == "fit":
            log.info(f"train_cuts: {self.hparams.train_cuts}")
            self.train_dataset = LhotseTTSDataset(
                self.text_tokenizer, self.model_config
            )
            log.info(f"train_dataset: {self.train_dataset}")
            self.train_sampler = DynamicCutSampler(
                self.hparams.train_cuts,
                max_duration=self.hparams.max_duration,
                shuffle=True,
                drop_last=True,
            )
            log.info(f"train_sampler: {self.train_sampler}")
        elif stage == "validate":
            log.info(f"val_cuts: {self.hparams.val_cuts}")
            self.val_dataset = LhotseTTSDataset(self.text_tokenizer, self.model_config)
            log.info(f"val_dataset: {self.val_dataset}")
            self.val_sampler = DynamicCutSampler(
                self.hparams.val_cuts,
                max_duration=self.hparams.max_duration,
                shuffle=False,
                drop_last=False,
            )
            log.info(f"val_sampler: {self.val_sampler}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            num_workers=self.train_num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            num_workers=self.val_num_workers,
            pin_memory=False,
        )


def my_prepare_libritts(corpus_dir: str = None):
    import os

    from lhotse.recipes.libritts import prepare_libritts

    if corpus_dir is None:
        corpus_dir = os.environ.get("LIBRITTS_CORPUS_DIR")

    if corpus_dir is None:
        raise ValueError("LIBRITTS_CORPUS_DIR is not set")

    ret = prepare_libritts(
        corpus_dir=corpus_dir,
        dataset_parts="all",
        output_dir="./data/libriTTS",
        num_jobs=8,
        link_previous_utt=False,
    )
    log.info(ret)

    return ret


if __name__ == "__main__":
    # run this in cli
    # MODEL_NAME_OR_PATH="/home/18T/zhouhao/models/Qwen2-0.5B" LIBRITTS_CORPUS_DIR="/home/18T/zhouhao/datasets/libriTTS/xzvf/LibriTTS" python lhoste_tts_dataset.py
    import os

    from lhotse.cut.set import CutSet

    model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH")
    if model_name_or_path is None:
        raise ValueError("MODEL_NAME_OR_PATH is not set")

    all_cuts = my_prepare_libritts()
    log.info(all_cuts)
    train_cuts = CutSet.from_manifests(**all_cuts["train-clean-100"])
    val_cuts = CutSet.from_manifests(**all_cuts["test-clean"])
    val_cuts = val_cuts.sample(32)

    train_cuts.describe()
    val_cuts.describe()

    text_tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)
    model_config = Qwen2Config.from_pretrained(model_name_or_path)

    data_module = LhotseDataModule(
        train_cuts=train_cuts,
        val_cuts=val_cuts,
        text_tokenizer=text_tokenizer,
        model_config=model_config,
    )

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    log.info(f"train_loader: {train_loader}")
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()
    log.info(f"val_loader: {val_loader}")
    cnt = 0
    for batch in val_loader:
        log.info(f"cnt: {cnt}")
        log.info(batch.keys())
        log.info(f"text_logits: {batch['text_logits'].shape}")
        log.info(f"wav: {batch['wav'].shape}")
        log.info(f"duration: {batch['duration'].shape}")
        cnt += 1
        if cnt > 10:
            break
