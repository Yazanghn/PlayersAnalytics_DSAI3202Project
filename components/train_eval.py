import argparse
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str)
    parser.add_argument("--test_input", type=str)
    parser.add_argument("--selected_features", type=str)
    parser.add_argument("--model_output", type=str)
    parser.add_argument("--metrics_output", type=str)
    args = parser.parse_args()

    train = pd.read_parquet(args.train_input)
    test = pd.read_parquet(args.test_input)

    with open(args.selected_features, "r") as f:
        selected = json.load(f)

    label = "season_market_value_eur"

    X_train, y_train = train[selected], train[label]
    X_test, y_test = test[selected], test[label]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    joblib.dump(model, args.model_output)

    metrics = {"rmse": rmse}
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
