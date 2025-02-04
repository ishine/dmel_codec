import hydra
import multiprocessing
import lightning.pytorch as pl

from omegaconf import DictConfig
import dmel_codec
from dmel_codec.utils.logger import RankedLogger
from dmel_codec.utils.print_config import print_config_tree
from dmel_codec.utils.utils import find_lastest_ckpt

dmel_root_path = dmel_codec.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path=f"{dmel_root_path}/config/lm", config_name="lm_config.yaml", version_base=None)
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

    logger.info("Instantiating tensorboard_logger...")
    tensorboard_logger = hydra.utils.instantiate(config.tensorboard_logger, _convert_="partial")

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks, 
        logger=tensorboard_logger, 
        _convert_="partial", 
        use_distributed_sampler=False, # Custom bucket sampler, the use_distributed_sampler need to be set to False
    )

    latest_ckpt_path = find_lastest_ckpt(config.get("codec_ckpt_dir", None))
    logger.info(f"start_training, latest_ckpt_path: {latest_ckpt_path}")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=latest_ckpt_path,
    )
    logger.info("training_finished")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(cfg)