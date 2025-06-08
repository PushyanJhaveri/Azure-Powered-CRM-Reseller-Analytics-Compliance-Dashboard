# Azure-Powered CRM Reseller Analytics & Compliance Dashboard

[![Azure ML SDK v2](https://img.shields.io/badge/Azure%20ML-SDK%20v2-blue)](https://learn.microsoft.com/azure/machine-learning/)  
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)  

---

## Project Overview

This end-to-end solution:

1. **Ingests** CRM & reseller data from Azure SQL Database 
2. **Cleans & features** it in Python (Pandas, custom utils)  
3. **Trains** Random Forest, XGBoost & HGB models in Azure ML  
4. **Registers & deploys** the best model as a real-time endpoint  
5. **Visualizes** commissions, compliance flags & forecasts in Tableau  

---

## Tech Stack

- **Language & Libraries:** Python 3.10, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib  
- **Cloud & MLOps:** Azure SQL, Azure Machine Learning (SDK v2), Azure Blob Storage, Managed Online Endpoints  
- **Visualization:** Tableau  
- **Dev & CI:** Jupyter Notebooks, VS Code

---

## Quickstart

1. **Clone repo**  
   ```bash
   git clone https://github.com/your-org/Azure-Powered-CRM-Reseller-Analytics-Compliance-Dashboard.git
   cd Azure-Powered-CRM-Reseller-Analytics-Compliance-Dashboard

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure Azure environment variables**
    ```bash
    export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
    export AZURE_RESOURCE_GROUP="<your-resource-group>"
    export AZURE_WORKSPACE_NAME="<your-workspace-name>"
    ```
4. **Training and register the model**
   ```bash
   python src/train.py
   ```
5. **Deploy the model as custom component**
   ```bash
   az ml job create --file azure/train_component.yml
   az ml online-endpoint invoke --name <endpoint-name> --request-file sample-request.json
   ```
6. **Test the deployed endpoint**
    ```bash
   az ml online-endpoint invoke \
     --name <your-endpoint-name> \
     --deployment blue \
     --request-file deploy/sample-request.json
    ```

---

## 📂 Repository Structure

```text
ece-commission-analytics/
│
├── README.md                       ← This file  
├── LICENSE                         ← Licenses  
├── .gitignore                      
├── requirements.txt                ← pip packages  
├── conda_env.yaml                  ← (optional) conda spec  
│
├── data/                           ← Sample data or pointers  
│   └── README.md                   ← How to fetch full datasets  
│
├── src/                            ← Production code & Azure ML scripts  
│   ├── train.py                    ← Pull → clean → featurize → train → register  
│   ├── score.py                    ← Inference script for your endpoint  
│   └── utils.py                    ← Shared cleaning & featurization helpers  
│
├── notebooks/                      ← Exploratory & demo notebooks  
│   ├── ECE_Commission_ML.ipynb     ← EDA & baseline Random Forest  
│   ├── ece_commission_tuning.ipynb ← Hyperparameter tuning  
│   ├── ece_commission_deployment.ipynb  
│   └── README.md                   ← Notebook descriptions  
│
├── azure/                          ← Azure ML component & pipeline specs  
│   ├── train_component.yml         ← Custom “train.py” component  
│   ├── score_component.yml         ← Custom “score.py” component  
│   └── pipeline.yml                ← (optional) end-to-end pipeline  
│
└── docs/                           ← Architecture diagrams, screenshots, write-ups  
    └── images/                     
        ├── model_performance.png  
        ├── feature_importance.png  
        └── deployment_architecture.png  
```
---

## 📂 Model Performance

| Model                        |   RMSE   |   MAE  |   R²  |
| ---------------------------- | :------: | :----: | :---: |
| **Random Forest (baseline)** | 2,790.29 | 305.29 | 0.849 |
| **Random Forest (tuned)**    | 2,650.1  | 290.4  | 0.865 |
| HistGradientBoosting         | 1,406.42 | 247.48 | 0.812 |
| XGBoost (tuned)              | 3,163.04 | 371.34 | 0.807 |


