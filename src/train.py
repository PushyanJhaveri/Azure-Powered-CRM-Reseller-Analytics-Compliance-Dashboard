# train.py
import os, argparse
import pandas as pd, numpy as np
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Azure ML SDK imports
from azureml.core import Run, Model, Dataset, Workspace

def load_raw_df_from_sql():
    server   = 'techentmarft2.database.windows.net'
    database = 'QAECECRM_Mar2025'
    username = 'dbadmin'
    password = 'DashTech1234'
    driver   = '{ODBC Driver 18 for SQL Server}'
    conn_str = (
        f"DRIVER={driver};SERVER={server};PORT=1433;"
        f"DATABASE={database};UID={username};PWD={password}"
    )
    cnxn = pyodbc.connect(conn_str)
    sql = """<paste your feature_sql here>"""  # your long SELECT…JOIN query
    return pd.read_sql(sql, cnxn)

def clean_and_featurize(df):
    # exactly your cleaning code from the notebook
    df = df.loc[:, ~df.columns.duplicated()]
    df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
    def to_seconds(t):
        tm = pd.to_datetime(t, format='%I:%M %p', errors='coerce')
        return tm.hour*3600 + tm.minute*60 if pd.notna(tm) else np.nan
    df['EventTimeSec'] = df['EventTime'].apply(to_seconds)
    for b in ['IsBackendDeal','IsSettingCovered','IsClosed']:
        if b in df: df[b] = df[b].astype('boolean')
    if 'EventBudget' in df:
        df['EventBudgetVal'] = (
          df['EventBudget'].astype(str)
            .str.replace(r'[^0-9\.]','',regex=True)
            .replace({'':np.nan})
            .pipe(pd.to_numeric, errors='coerce')
        )
    df['ECECommission'] = df['ECECommission'].fillna(0)
    nums = df.select_dtypes(include='number').columns.drop('ECECommission')
    for c in nums: df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include='object'): df[c] = df[c].fillna("Unknown")
    grp = [c for c in df.columns if c not in ('ECECommission','EventDate')]
    df = df.groupby(grp, dropna=False)['ECECommission'].sum().reset_index()
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, help="TabularDataset name or ID")
    parser.add_argument("--model_output",   type=str, default="./outputs", help="folder to write model")
    args = parser.parse_args()

    # Get the Azure ML run context
    run = Run.get_context()
    ws  = run.experiment.workspace

    # 1) Load raw data
    if args.input_dataset:
        ds = Dataset.get_by_name(ws, args.input_dataset)
        df = ds.to_pandas_dataframe()
    else:
        df = load_raw_df_from_sql()

    # 2) Clean & featurize
    df_clean = clean_and_featurize(df)

    # 3) Split & train
    y = df_clean.pop('ECECommission')
    X = pd.get_dummies(df_clean.drop(columns=['EventBudget','EventTime',
                                              'PresenterOrg','VenueCity',
                                              'ResellerCity','EventTypeName']),
                       drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 4) Save model
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "rf_model.joblib")
    joblib.dump(rf, model_path)
    run.upload_file(name="rf_model.joblib", path_or_stream=model_path)

    # 5) Register model
    Model.register(workspace=ws,
                   model_path="rf_model.joblib",    # this file in outputs
                   model_name="ece_commission_rf",
                   tags={"method":"RandomForest"},
                   description="RF predicting ECECommission")

    run.complete()

if __name__=="__main__":
    main()
