import argparse
import pandas as pd
import json
from sklearn.feature_selection import VarianceThreshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str)
    parser.add_argument("--selected_features_output", type=str)
    args = parser.parse_args()

    df = pd.read_parquet(args.train_input)

    label = "season_market_value_eur"
    X = df.drop(columns=[label])
    y = df[label]

    selector = VarianceThreshold()
    selector.fit(X)

    mask = selector.get_support()
    selected = list(X.columns[mask])

    with open(args.selected_features_output, "w") as f:
        json.dump(selected, f)

    print("Selected feature count:", len(selected))

if __name__ == "__main__":
    main()
