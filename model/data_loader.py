import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm

class DTIDataset(Dataset):
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

        #truncate if longer than max length - NEED TO OPTIMIZE THIS LATER
        pr = pr[:self.seqlen]
        lig = lig[:self.smilelen]

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
