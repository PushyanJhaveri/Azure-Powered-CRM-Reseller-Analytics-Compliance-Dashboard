# src/score.py
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Score ECE Commission model")
    parser.add_argument("--model‐path", type=str, required=True, help="Path to saved model (joblib or MLflow folder)")
    parser.add_argument("--input", type=str, required=False, help="JSON string or CSV row of features to score")
    parser.add_argument("--input‐file", type=str, required=False, help="Path to a JSON file containing features")
    return parser.parse_args()

def load_model(model_path):
    # If MLflow folder, you can do: mlflow.pyfunc.load_model(model_path)
    return joblib.load(os.path.join(model_path, "rf_model.joblib"))

def preprocess_input(raw):
    # raw might be a JSON string or list of features in the notebook 
    if isinstance(raw, str):
        data = json.loads(raw)
        X = pd.DataFrame(data["data"], columns=data["columns"])
    else:
        # assume CSV row:
        values = list(map(float, raw.split(",")))
        X = pd.DataFrame([values])
    return X

def main():
    args = parse_args()
    model = load_model(args.model_path)

    if args.input_file:
        with open(args.input_file, "r") as f:
            req = json.load(f)
        X = pd.DataFrame(req["data"], columns=req["columns"])
    elif args.input:
        X = preprocess_input(args.input)
    else:
        raise ValueError("Provide either --input or --input‐file")

    # Ensuring columns align with training order
  
    y_pred = model.predict(X)
    print(json.dumps({"predictions": y_pred.tolist()}))

if __name__ == "__main__":
    main()
