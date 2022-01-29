import os
import warnings

warnings.filterwarnings('ignore')
import argparse
import glob

import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_curve

from src.technical_utils import load_obj
from src.utils import set_seed


def find_threshold_if_needed(labels, probas):
    fpr, tpr, thlds = roc_curve(labels, probas)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thlds[optimal_idx]
    acc = accuracy_score(labels, [int(p > threshold) for p in probas])
    print(f"Threshold: {threshold}, Validation Accuracy: {acc}")
    return threshold, acc


def fit_bin_threshold(val_df):
    threshold, validation_accuracy = find_threshold_if_needed(
        labels=val_df['label'].values,
        probas=val_df['probs'].values,
    )
    val_df['y_pred_best'] = val_df['probs'].apply(lambda x: int(x > threshold))
    return val_df


def make_prediction(cfg: DictConfig) -> pd.DataFrame:
    """
    Run pytorch-lightning model inference

    Args:
        cfg: hydra config

    Returns:
        None
    """
    set_seed(cfg.training.seed)
    model_names = glob.glob(f'../outputs/{cfg.test.run_name}/saved_models/*')
    best_model = [name for name in model_names if 'best' in name][0]

    # load from checkpoint
    tuner = load_obj(cfg.training.wrapper_name)(cfg=cfg)
    model = tuner.model
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint)
    model.to(cfg.test.device)
    model.eval()

    # get val dataloader
    cfg.datamodule.batch_size = cfg.test.batch_size
    dm = load_obj(cfg.datamodule.data_module_name)(cfg=cfg)
    dm.prepare_data()
    dm.setup()
    val_df = dm.val_data
    val_dataloader = dm.val_dataloader()

    # set bin threshold
    if cfg.test.threshold:
        threshold = torch.tensor([cfg.test.threshold]).to(cfg.test.device)
    else:
        threshold = torch.tensor([0.5]).to(cfg.test.device)

    # labels mapping
    labels_map = cfg.test.labels_map
    labels_map_inv = {v: k for k, v in labels_map.items()}

    probs, preds = [], []
    with torch.no_grad():
        for _, batch in enumerate(val_dataloader):
            batch = {k: v.to(cfg.test.device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch['token_type_ids'],
            )
            y_prob = torch.sigmoid(logits).squeeze(1)
            y_pred = (y_prob > threshold).float()

            probs.extend(y_prob.detach().cpu().numpy())
            preds.extend(y_pred.detach().cpu().numpy())

    val_df['probs'] = probs
    val_df['y_pred'] = preds
    val_df['label_pred'] = val_df['y_pred'].apply(lambda x: labels_map[x])
    val_df['label'] = val_df['label'].apply(lambda x: labels_map_inv[x])

    acc = accuracy_score(val_df['y_pred'], val_df['label'])
    print(f"validation accuracy: {acc}, threshold: {threshold.item()}")

    return val_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model estimation on a val dataset')
    parser.add_argument(
        '--task_name',
        help='name of task and dataset',
        type=str,
        choices=['terra', 'russe'],
        default='terra'
    )
    parser.add_argument(
        '--device',
        help='inference device',
        type=str,
        default='cuda'
    )
    args = parser.parse_args()

    initialize(config_path="../cfg/inference/")
    test_cfg = compose(config_name="test")
    test_cfg = test_cfg[args.task_name]
    test_cfg['device'] = args.device

    path = f'../outputs/{test_cfg.run_name}/.hydra/config.yaml'
    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)
        cfg_yaml['test'] = test_cfg
        cfg = OmegaConf.create(cfg_yaml)
    print(OmegaConf.to_yaml(cfg))

    # predict labels
    val_df_with_labels = make_prediction(cfg)
    val_df_with_labels = fit_bin_threshold(val_df_with_labels)
    # save results
    save_dir = f'../outputs/{args.task_name}_predicts/'
    if not os.path.exists(save_dir):  # type: ignore
        os.makedirs(save_dir, exist_ok=True)

    val_df_with_labels.to_csv(
        os.path.join(save_dir, f'{cfg.test.run_name}_val_preds.csv'),
        sep='\t',
        index=False
    )
