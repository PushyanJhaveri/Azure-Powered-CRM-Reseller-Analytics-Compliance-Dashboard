# src/train.py
import os
import argparse
import pandas as pd, 
import numpy as np
import pyodbc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
# Azure ML SDK imports
from azureml.core import Run, Model, Dataset, Workspace

def parse_args():
    parser = argparse.ArgumentParser(description="Train ECE Commission RF and register model")
    parser.add_argument("--connection‐string", type=str, required=True, help="ODBC connection string")
    parser.add_argument("--model‐output‐path", type=str, default="outputs/model", help="Where to save the trained model")
    parser.add_argument("--subscription‐id", type=str, required=True)
    parser.add_argument("--resource‐group", type=str, required=True)
    parser.add_argument("--workspace‐name", type=str, required=True)
    return parser.parse_args()

conn_str = (
        f"DRIVER={driver};SERVER={server};PORT=1433;"
        f"DATABASE={database};UID={username};PWD={password}"
    )

def pull_data(conn_str):
    sql = """
    SELECT
      ca.ECECommission, ca.Gross, ca.ContractId, ca.ArtistId,
      c.AgentId, c.PresenterId, c.VenueId, c.LineOfBusinessId,
      c.ContractStatusId, c.IsBackendDeal, c.EventTypeId,
      v.Capacity AS VenueCapacity, v.PhysicalCity AS VenueCity,
      v.IsSettingCovered, v.PhysicalGeoLatitude,
      v.PhysicalGeoLongitude, b.EventBudget, b.EventTime,
      b.IsClosed, b.ClosedReasonId, ced.EventDate,
      r.MailingCity AS ResellerCity, r.MailingStateId AS ResellerStateId,
      et.Name AS EventTypeName, p.OrganizationName AS PresenterOrg,
      ac.OriginalAmount, act.TransactionAmount,
      ap.LineOfBusinessId AS LineOfBusinessId, lp.EcePercentage,
      lp.ArtistPercentage, ot.ECEProcurementFee, c.ECECommissionRate
    FROM ContractArtist ca
    LEFT JOIN Contract c ON ca.ContractId = c.ContractId
    LEFT JOIN Venue v ON c.VenueId = v.VenueId
    LEFT JOIN BlueCard b ON c.BlueCardId = b.BlueCardId
    LEFT JOIN ContractEventDate ced ON c.ContractId = ced.ContractId
    LEFT JOIN Reseller r ON c.ResellerId = r.ResellerId
    LEFT JOIN LuEventType et ON c.EventTypeId = et.EventTypeId
    LEFT JOIN Presenter p ON c.PresenterId = p.PresenterId
    LEFT JOIN ArtistCharge ac ON ca.ContractArtistId = ac.ArtistId
    LEFT JOIN ArtistChargeTransaction act ON ac.ArtistChargeId = act.ArtistChargeId
    LEFT JOIN ArtistProgram ap ON ca.ArtistId = ap.ArtistId
    LEFT JOIN LuProgram lp ON ap.ProgramId = lp.ProgramId
    LEFT JOIN OfferTerms ot ON ca.ArtistId = ot.ArtistId
    """
    cnxn = pyodbc.connect(conn_str)
    df = pd.read_sql(sql, cnxn)
    return df


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
   
    # 6) Register to Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    register_model(ml_client, args.model_output_path)

    run.complete()

if __name__=="__main__":
    main()
