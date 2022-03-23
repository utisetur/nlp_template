import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
import warnings
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.technical_utils import load_obj, save_useful_info, set_seed

warnings.filterwarnings("ignore")


def learning_rate_search(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    run_name = os.path.basename(os.getcwd())
    cfg.callbacks.model_checkpoint.params.dirpath = Path(
        os.getcwd(), cfg.callbacks.model_checkpoint.params.dirpath
    ).as_posix()

    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            if "experiment_name" in logger.params.keys():
                logger.params["experiment_name"] = run_name
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))
    callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint.params))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    dm = load_obj(cfg.datamodule.data_module_name)(cfg=cfg)
    dm.prepare_data()
    dm.setup()
    train_dataloader_size = len(dm.train_dataloader())

    model = load_obj(cfg.training.trainer_name)(
        cfg=cfg, train_dataloader_size=train_dataloader_size
    )

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(
        model=model,
        datamodule=dm,
        min_lr=8e-8,
        max_lr=10,
        num_training=500,
    )
    # Results
    res = lr_finder.results
    print("Loss | lr")
    print(sorted(zip(res["loss"], res["lr"])))

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print("best lr:", new_lr)


@hydra.main(config_path="../cfg", config_name="rte_config")
def run_learning_rate_search(cfg: DictConfig) -> None:
    os.makedirs("../logs", exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info()
    learning_rate_search(cfg)


if __name__ == "__main__":
    """
    Example:
    python bin/train.py --config-name='config'
    """
    run_learning_rate_search()
