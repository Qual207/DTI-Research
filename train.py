"""Train the DeepDTA model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import model.net as net
import model.data_loader as data_loader
import utils
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/processed/davis.csv',
                    help="Path to dataset CSV file")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json and saved weights")
parser.add_argument('--restore_file', default=None,
                    help="Optional — checkpoint file to restore from (without extension)")
args = parser.parse_args()


def train(model, optimizer, loss_fn, dataloader, params):
    """Train for one epoch"""
    model.train()
    loss_avg = utils.RunningAverage()
    
    with tqdm(total=len(dataloader)) as t:
        for batch in dataloader:
            pr, lig, y = batch['protein'], batch['ligand'], batch['affinity']
            if params.cuda:
                pr, lig, y = pr.cuda(non_blocking=True), lig.cuda(non_blocking=True), y.cuda(non_blocking=True)

            optimizer.zero_grad()
            pred = model(pr, lig)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())
            t.set_postfix(loss=f"{loss_avg():05.3f}")
            t.update()
    
    return loss_avg()


def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch"""
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info(f"Restoring parameters from {restore_path}")
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = float('inf')

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch+1}/{params.num_epochs}")
        train_loss = train(model, optimizer, loss_fn, train_loader, params)
        val_metrics = evaluate(model, loss_fn, val_loader, params)

        val_loss = val_metrics['loss']
        is_best = val_loss <= best_val_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        if is_best:
            logging.info(f"- Found new best model (val_loss={val_loss:.4f})")
            best_val_loss = val_loss
            utils.save_dict_to_json(val_metrics, os.path.join(model_dir, "metrics_val_best.json"))

        utils.save_dict_to_json(val_metrics, os.path.join(model_dir, "metrics_val_last.json"))

    logging.info("Training complete!")


if __name__ == "__main__":
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading datasets...")
    df = pd.read_csv(args.data_path)

    train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    val_idx, _ = train_test_split(temp_idx, test_size=0.5, random_state=42)

    dataset = data_loader.DTIDataset(df)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=params.batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=params.batch_size, collate_fn=data_loader.collate_fn)
    logging.info("- Done loading data.")

    model = net.DeepDTA(len(dataset.protein_vocab)+1, len(dataset.ligand_vocab)+1,
                        channel=params.channel,
                        protein_kernel_size=params.protein_kernel_size,
                        ligand_kernel_size=params.ligand_kernel_size)
    if params.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    loss_fn = nn.MSELoss()

    logging.info(f"Starting training for {params.num_epochs} epochs")
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, params, args.model_dir, args.restore_file)
