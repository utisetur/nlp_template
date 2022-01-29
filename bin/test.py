import json
import os
import warnings

warnings.filterwarnings('ignore')
import argparse
import glob
import warnings

import pandas as pd
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.technical_utils import load_obj
from src.utils import set_seed

warnings.filterwarnings('ignore')
import torch


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

    tuner = load_obj(cfg.training.wrapper_name)(cfg=cfg)
    model = tuner.model
    # model = load_obj(cfg.model.class_name)(cfg=cfg)
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint)
    model.to(cfg.test.device)
    model.eval()

    dm = load_obj(cfg.datamodule.data_module_name)(cfg=cfg)
    dm.prepare_data()
    dm.setup(stage='test')
    test_df = dm.test_data
    test_dataloader = dm.test_dataloader()

    threshold = torch.tensor([cfg.test.threshold]).to(cfg.test.device)
    labels_map = cfg.test.labels_map
    probs, preds = [], []
    with torch.no_grad():
        for ind, batch in enumerate(test_dataloader):
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

    test_df['probs'] = probs
    test_df['y_pred'] = preds
    test_df['label_pred'] = test_df['y_pred'].apply(lambda x: labels_map[x])

    return test_df


def make_submit(df, path_to_save: str):
    """
    File example:
    {"idx": 2, "label": "false"}
    {"idx": 3, "label": "true"}
    """
    submit_file = []
    for i, row in df.iterrows():
        submit_file.append({"idx": row.idx, "label": row.label_pred})

    with open(path_to_save, 'w') as f:
        for item in submit_file:
            f.write(json.dumps(item) + "\n")

    print(f'predictions saved to: {path_to_save}')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make submission')
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
    test_df_with_labels = make_prediction(cfg)

    # save results
    save_dir = f'../outputs/{args.task_name}_predicts/'
    if not os.path.exists(save_dir):  # type: ignore
        os.makedirs(save_dir, exist_ok=True)

    test_df_with_labels.to_csv(
        os.path.join(save_dir, f'{cfg.test.run_name}_test_preds.csv'),
        sep='\t',
        index=False
    )
    # save submission file
    make_submit(test_df_with_labels, path_to_save=os.path.join(save_dir, f'{args.task_name}.jsonl'))
