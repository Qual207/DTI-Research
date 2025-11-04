import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm



class Dataset(Dataset):
    def __init__(self, df, seqlen=2000, smilen=200):
        self.proteins = df['proteins'].values
        self.ligands = df['ligands'].values
        self.affinity = df['affinity'].values
        self.smilelen = smilen
        self.seqlen = seqlen

        self.protein_vocab = set()
        self.ligand_vocab = set()
        for prot in self.proteins:
            self.protein_vocab.update(list(prot))
        for lig in self.ligands:
            self.ligand_vocab.update(list(lig))

        # having a dummy token to pad the sequences to the max length
        self.protein_dict = {pro_token: i+1 for i, pro_token in enumerate(self.protein_vocab)}
        self.ligand_dict = {lig_token: i+1 for i, lig_token in enumerate(self.ligand_vocab)}
        

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        pr = self.proteins[idx]
        lig = self.ligands[idx]
        target = self.affinity[idx]

        protein_ids = [self.protein_dict[x] for x in pr] + [0] * (self.seqlen - len(pr))
        ligand_ids = [self.ligand_dict[x] for x in lig] + [0] * (self.smilelen - len(lig))

        return {
            'protein': torch.tensor(protein_ids, dtype=torch.long),
            'ligand': torch.tensor(ligand_ids, dtype=torch.long),
            'affinity': torch.tensor(target, dtype=torch.float)
        }

def collate_fn(batch):
    proteins = torch.stack([x['protein'] for x in batch])
    ligands = torch.stack([x['ligand'] for x in batch])
    affinity = torch.stack([x['affinity'] for x in batch])
    return proteins, ligands, affinity


class Trainer:
    def __init__(self, model, df, train_idx, val_idx, test_idx, protein_kernel=8, ligand_kernel=6, channel=32, seqlen=2000, smilen=200, log_file='train.log'):
        self.dataset = Dataset(df, seqlen=seqlen, smilen=smilen)
        self.protein_vocab_size = len(self.dataset.protein_vocab) + 1
        self.ligand_vocab_size = len(self.dataset.ligand_vocab) + 1
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)
        self.protein_kernel = protein_kernel
        self.ligand_kernel = ligand_kernel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model(
            self.protein_vocab_size,
            self.ligand_vocab_size,
            channel,
            protein_kernel,
            ligand_kernel
        ).to(self.device)
        
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def train(self, lr, num_epochs, batch_size, save_path):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        writer = SummaryWriter()

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last = False, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)

        with open('protein_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.protein_dict, f)
        with open('ligand_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.ligand_dict, f)

        
        best_weights = self.model.state_dict()
        best_val_loss = np.inf
        best_epoch = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            with tqdm(total=len(train_loader)) as pbar:
                for protein, ligand, target in train_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    pbar.update(1)

            train_loss /= len(train_loader)
            self.logger.info('Epoch: {} - Training Loss: {:.6f}'.format(epoch+1, train_loss))
            writer.add_scalar('train_loss', train_loss, epoch)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for protein, ligand, target in val_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = deepcopy(self.model.state_dict())
                best_epoch = epoch
                self.logger.info('Best Model So Far in Epoch: {}'.format(epoch+1))
            self.logger.info('Epoch: {} - Validation Loss: {:.6f}'.format(epoch+1, val_loss))
            writer.add_scalar('val_loss', val_loss, epoch)
        
        self.model.load_state_dict(best_weights)
        test_result = []
        with torch.no_grad():
            for protein, ligand, target in test_loader:
                protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                output = self.model(protein, ligand)
                test_result.append(output.cpu().numpy())
        test_result = np.concatenate(test_result)
        np.savetxt('test-result-prk{}-ldk{}.txt'.format(self.protein_kernel, self.ligand_kernel), test_result)
        
        self.logger.info('Best Model Loaded from Epoch: {}'.format(best_epoch+1))
        torch.save(self.model.state_dict(), save_path)
        self.logger.handlers[0].close()
        self.logger.removeHandler(self.logger.handlers[0])
        writer.close()
    
