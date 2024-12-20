import hydra
import multiprocessing
from typing import List

import lightning.pytorch as pl
import rootutils
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from dmel_codec.utils.logger import RankedLogger
from dmel_codec.utils.print_config import print_config_tree
from dmel_codec.utils.utils import find_lastest_ckpt
from dmel_codec.utils.instantiators import instantiate_loggers


rootutils.setup_root(__file__, indicator=".dMel_chatMusic_root", pythonpath=True)

logger = RankedLogger(__name__, rank_zero_only=True)


# TODO wzy_config/zh_config/lc_config 文件夹的路径
@hydra.main(config_path="config", config_name="dMel_example", version_base=None)
def main(config: DictConfig) -> None:
    print_config_tree(config)

    pl.seed_everything(config.seed)

    logger.info(f"Instantiating datamodule <{config.data._target_}>.")
    datamodule = hydra.utils.instantiate(config.data, _convert_="partial")

    logger.info(f"Instantiating model <{config.model._target_}>.")
    model = hydra.utils.instantiate(config.model, _convert_="partial")

    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating datamodule <{cb_conf._target_}>.")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    logger.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(config.get("logger"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    latest_ckpt_path = find_lastest_ckpt(config.codec_ckpt_dir)
    logger.info(f"start_training, latest_ckpt_path: {latest_ckpt_path}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=latest_ckpt_path)
    logger.info("training_finished")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
