import numpy as np
import pandas as pd
import json
import pickle
import os

def deepdta_to_csv(dataset_dir, output_file):
    #load the binding affinity matrix (Y)
    Y = pickle.load(open(os.path.join(dataset_dir, "Y"), "rb"))
    ligands_path = os.path.join(dataset_dir, "ligands_iso.txt")
    proteins_path = os.path.join(dataset_dir, "proteins.txt")

    #load ligands and proteins
    with open(ligands_path, "r") as f:
        ligands_dict = json.load(f)
    ligand_ids = list(ligands_dict.keys())
    ligands = [ligands_dict[lig_id] for lig_id in ligand_ids]

    with open(proteins_path, "r") as f:
        proteins_dict = json.load(f)
    protein_ids = list(proteins_dict.keys())
    proteins = [proteins_dict[prot_id] for prot_id in protein_ids]
    
    #find non-NaN pairs
    drug_inds, protein_inds = np.where(np.isnan(Y)==False)

    data = []
    for d_idx, p_idx in zip(drug_inds, protein_inds):
            data.append({
                "drug_id": ligand_ids[d_idx],              
                "protein_id": protein_ids[p_idx],         
                "ligands": ligands[d_idx],            
                "proteins": proteins[p_idx],      
                "affinity": Y[d_idx, p_idx]               
            })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV saved to {output_file}")
