from typing import Any, Dict

import torch
from tqdm import tqdm


def batch2device(batch: Dict, device: torch.device) -> Dict:
    for key, value in batch.items():
        batch[key] = value.to(device)
    return batch


def get_test_scores(model, test_dataloader):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.set_grad_enabled(False)
    model.to(device)
    model.eval()

    test_preds = []
    test_probs = []
    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            batch = batch2device(batch, device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            y_prob, y_pred = torch.max(probs, dim=1)
            test_preds.extend(y_pred.numpy())
            test_probs.extend(probs.numpy())
    return test_preds, test_probs


def make_submission_file(test_dataset, test_preds):
    """
    File example:
    {"idx": 12, "label": "not_entailment"}
    {"idx": 13, "label": "entailment"}
    """
    tag2label = {v: k for k, v in test_dataset.labels_map.items()}
    submit_file = []
    for idx in [i[-1] for i in test_dataset.data]:
        submit_file.append({"idx": idx, "label": tag2label[test_preds[idx]]})

    return submit_file


def freeze_until(net: Any, param_name: str = None) -> None:
    """
    Freeze net until param_name

    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD

    Args:
        net:
        param_name:

    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
