import numpy as np
import torch
from torch.nn import functional as F
from transformers import Trainer

from baseline.metrics import macro_f1


def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Numerically stable sigmoid focal loss, mean over samples and labels."""
    if logits.shape != targets.shape or gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid focal loss inputs")
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


class FocalTrainer(Trainer):
    def __init__(self, *args, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        # Loss averages the micro-batch itself; Trainer applies accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = binary_focal_loss(outputs.logits, labels, self.focal_alpha, self.focal_gamma)
        return (loss, outputs) if return_outputs else loss


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, labels, task, max_length, include_labels=True):
        if not rows:
            raise ValueError("Dataset must contain at least one sample")
        self.encodings = tokenizer([r["input"] for r in rows], truncation=True, max_length=max_length)
        self.label_ids = {label: i for i, label in enumerate(labels)}
        self.rows, self.task, self.include_labels = rows, task, include_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.include_labels:
            target = torch.zeros(len(self.label_ids), dtype=torch.float32)
            label_id = self.label_ids.get(self.rows[index][self.task])
            if label_id is not None:
                target[label_id] = 1
            # All-zero gold represents an unseen validation class; it is never a positive.
            # Tokenizer padding accepts nested Python lists; a list of tensors
            # cannot be converted by BatchEncoding on all supported versions.
            item["labels"] = target.tolist()
        return item


def compute_metrics(evaluation, true_labels=None, labels=None):
    logits, one_hot = evaluation.predictions, evaluation.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    truth = np.where(one_hot.sum(axis=1) > 0, one_hot.argmax(axis=1), -1)
    predicted = logits.argmax(axis=1)
    k = min(5, logits.shape[1])
    top = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    f1 = macro_f1(true_labels, [labels[int(i)] for i in predicted]) if true_labels is not None else macro_f1(truth, predicted)
    return {"macro_f1": f1,
            "recall_at_5": float(np.mean(np.any(top == truth[:, None], axis=1)))}
