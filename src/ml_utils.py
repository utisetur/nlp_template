from typing import Any, Dict

import torch
from tqdm import tqdm


def batch2device(batch: Dict, device: torch.device) -> Dict:
    for key, value in batch.items():
        batch[key] = value.to(device)
    return batch


def get_test_scores(model, test_dataloader):
    """
    File example:
    {"idx": 12, "label": "not_entailment"}
    {"idx": 13, "label": "entailment"}
    """
    label2tag = test_dataloader.dataset.labels_map
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.set_grad_enabled(False)
    model.to(device)
    model.eval()

    submit_file = []
    scores = []
    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            idx = batch["idx"].numpy()
            batch = batch2device(batch, device)
            logits = model(**batch)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            y_prob, y_pred = torch.max(probs, dim=1)
            y_pred = y_pred.numpy()
            scores.extend(list(zip(idx, y_pred)))

    for s in scores:
        submit_file.append({"idx": int(s[0]), "label": str(label2tag[s[1]])})

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
