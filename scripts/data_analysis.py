import pandas as pd
from statistics import mean

def analyze_dataset(df, name):
    print(f"Current Dataset: {name.upper()}")

    print(f"Number of rows (samples): {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:", list(df.columns))
    print("\nMissing values:\n", df.isnull().sum())

    print("\nExample rows:\n", df.head(3))

    print("\nAffinity stats:")
    print(df["affinity"].describe())

    avg_drug_len = mean(df["ligands"].apply(len))
    print(f"\nAverage drug SMILES length: {avg_drug_len:.2f}")

    avg_prot_len = mean(df["proteins"].apply(len))
    print(f"Average protein sequence length: {avg_prot_len:.2f}")

    print("\nEND OF ANALYSIS\n")


davis_df = pd.read_csv("data/processed/davis.csv")
kiba_df = pd.read_csv("data/processed/kiba.csv")

analyze_dataset(davis_df, "Davis")
analyze_dataset(kiba_df, "KIBA")
