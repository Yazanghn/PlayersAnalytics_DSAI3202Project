import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_gold_folder(path):
    """
    Loads all parquet files from the GOLD folder.
    The folder contains:
        - player_season_features (delta)
        - player_season_value_features (delta)
    We load both and merge automatically.
    """

    print("Loading GOLD folder:", path)

    # Detect subfolders
    subdirs = [d for d in os.listdir(path) if not d.startswith(".")]

    if len(subdirs) == 0:
        raise ValueError("No subdirectories found inside GOLD folder.")

    # Paths
    feat_dir = os.path.join(path, "player_season_features")
    value_dir = os.path.join(path, "player_season_value_features")

    print(" Features directory:", feat_dir)
    print(" Value directory:", value_dir)

    # Load the parquet files (delta logs ignored automatically)
    features_df = pd.read_parquet(feat_dir)
    value_df = pd.read_parquet(value_dir)

    print("Loaded features_df shape:", features_df.shape)
    print("Loaded value_df shape:", value_df.shape)

    # Merge the two dataframes
    merged = pd.merge(
        features_df,
        value_df,
        on=["player_id", "season"],
        how="inner"
    )

    print("Merged GOLD dataframe shape:", merged.shape)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_data", type=str)
    parser.add_argument("--train_output", type=str)
    parser.add_argument("--test_output", type=str)
    args = parser.parse_args()

    print("Starting Feature Retrieval Component")

    # Load GOLD layer & merge
    df = load_gold_folder(args.gold_data)

    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print("Training set:", train_df.shape)
    print("Test set:", test_df.shape)

    # Save outputs
    train_df.to_parquet(args.train_output, index=False)
    test_df.to_parquet(args.test_output, index=False)

    print("Saved train to:", args.train_output)
    print("Saved test to:", args.test_output)
    print("Feature Retrieval DONE")


if __name__ == "__main__":
    main()
    
