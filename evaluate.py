import torch
import numpy as np
import logging

def evaluate(model, loss_fn, dataloader, params):
    """Compute loss on validation/test set"""
    model.eval()
    losses = []
    with torch.no_grad():
        for pr, lig, y in dataloader:
            if params.cuda:
                pr, lig, y = pr.cuda(non_blocking=True), lig.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = model(pr, lig)
            loss = loss_fn(pred, y)
            losses.append(loss.item())

    avg_loss = np.mean(losses)
    metrics = {'loss': avg_loss}
    logging.info(f"- Validation loss: {avg_loss:.4f}")
    return metrics
