import argparse
import glob
import warnings

import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorWithPadding

from src.technical_utils import load_obj
from src.utils import set_seed

warnings.filterwarnings('ignore')
import pandas as pd
import torch

from src.get_dataset import get_test_dataset


def run_inference(cfg: DictConfig, sentence1: str, sentence2: str) -> None:
    """
    Run pytorch-lightning model inference

    Args:
        cfg: hydra config

    Returns:
        None
    """
    set_seed(cfg.training.seed)
    model_names = glob.glob(f'../outputs/{cfg.inference.run_name}/saved_models/*')
    best_model = [name for name in model_names if 'best' in name][0]

    model = load_obj(cfg.model.class_name)(cfg=cfg)
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint)
    model.to(cfg.inference.device)
    model.eval()

    tokenizer = load_obj(cfg.datamodule.tokenizer).from_pretrained(cfg.datamodule.pretrained_tokenizer)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    request = [[sentence1, sentence2]]
    request_dataset = get_test_dataset(cfg, request, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(
        request_dataset,
        batch_size=len(request_dataset),
        num_workers=1,
        pin_memory=False,
        collate_fn=collator,
        shuffle=False,
    )

    threshold = torch.tensor([cfg.inference.threshold]).to(cfg.inference.device)
    labels_map = cfg.inference.labels_map
    probs, preds = [], []
    with torch.no_grad():
        for ind, batch in enumerate(test_dataloader):
            batch = {k: v.to(cfg.inference.device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch['token_type_ids'],
            )
            y_prob = torch.sigmoid(logits).squeeze(1)
            y_pred = (y_prob > threshold).int()

            probs.extend(y_prob.detach().cpu().numpy())
            preds.extend(y_pred.detach().cpu().numpy())

    res = pd.DataFrame.from_dict({'sentence1': [sentence1], 'sentence2': [sentence2]})
    res['probs'] = probs
    res['y_pred'] = preds
    res['label'] = res['y_pred'].apply(lambda x: labels_map[x])
    print(f'predicted label: {res["label"].values[0]}, prob: {round(res["probs"].values[0], 3)}')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make model inference')
    parser.add_argument(
        '--task_name',
        help='task name',
        type=str,
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
    inference_cfg = compose(config_name="inference")
    inference_cfg = inference_cfg[args.task_name]
    inference_cfg['device'] = args.device

    path = f'../outputs/{inference_cfg.run_name}/.hydra/config.yaml'
    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)
        cfg_yaml['inference'] = inference_cfg
        cfg = OmegaConf.create(cfg_yaml)

    print('Task name:', args.task_name)
    sentence1 = input("Введите sentence1: ")
    sentence2 = input("Введите sentence2: ")

    run_inference(cfg, sentence1=sentence1, sentence2=sentence2)
