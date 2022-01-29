import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
import warnings
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.technical_utils import convert_to_jit, load_obj
from src.utils import save_useful_info, set_seed

warnings.filterwarnings('ignore')


def run(cfg: DictConfig) -> None:
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
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
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

    model = load_obj(cfg.training.wrapper_name)(cfg=cfg, train_dataloader_size=train_dataloader_size)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model and cfg.general.save_best:
        if os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder
            save_name = os.path.basename(os.path.normpath(best_path))
            model = model.load_from_checkpoint(
                best_path, cfg=cfg, train_dataloader_size=train_dataloader_size, strict=False
            )
            model_name = Path(
                cfg.callbacks.model_checkpoint.params.dirpath, f'best_{save_name}'.replace('.ckpt', '.pth')
            ).as_posix()
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)

    if cfg.general.convert_to_jit and os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
        best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
        save_name = os.path.basename(os.path.normpath(best_path))
        # TODO: check convert_to_jit func
        convert_to_jit(model, save_name, cfg)


@hydra.main(config_path='../cfg', config_name='ruroberta_terra_with_cls_head_config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('../logs', exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    """
    Example:
    python bin/ruroberta_train.py --config-name='ruroberta_terra_config'
    python bin/ruroberta_train.py --config-name='ruroberta_terra_with_cls_head_config'
    """
    run_model()
