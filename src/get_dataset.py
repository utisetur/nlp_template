from typing import List

from omegaconf import DictConfig

from src.technical_utils import load_obj


def get_test_dataset(cfg: DictConfig, test_data: List, tokenizer):
    """
    Get test dataset
    :param cfg:
    :param test_data:
    :param tokenizer:
    :return:
    """
    dataset_class = load_obj(cfg.inference.dataset_class)
    test_dataset = dataset_class(
        data=test_data,
        tokenizer=tokenizer,
    )
    return test_dataset
