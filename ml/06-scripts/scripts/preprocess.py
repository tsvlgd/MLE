import argparse
import pandas as pd
from utils.data_helper import load_data

def preprocess_data(input_file: str, output_file: str) -> None:
    """Preprocess data: handle missing values, normalize, and save."""
    df = load_data(input_file)
    df = df.dropna().apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.dtype in ['int64', 'float64'] else x)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()
    preprocess_data(args.input, args.output)


# usage: python scripts/preprocess.py --input data/raw.csv --output data/processed.csv