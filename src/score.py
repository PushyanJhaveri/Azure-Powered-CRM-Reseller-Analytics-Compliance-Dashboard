# score.py
import json, joblib, os
import pandas as pd
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("ece_commission_rf")
    model = joblib.load(model_path)

def clean_and_featurize(df):
    # copy the same cleaning code from train.py’s function
    # … (parse dates, fillna, one‐hot etc.)
    # make sure it matches exactly
    return df

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df   = pd.DataFrame.from_dict(data)
        df_clean = clean_and_featurize(df)
        X = pd.get_dummies(df_clean.drop(columns=['EventBudget','EventTime',
                                                  'PresenterOrg','VenueCity',
                                                  'ResellerCity','EventTypeName']),
                           drop_first=True)
        # align columns to training
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        preds = model.predict(X)
        return json.dumps(preds.tolist())
    except Exception as e:
        return json.dumps({"error": str(e)})
