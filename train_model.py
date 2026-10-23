import os
import requests
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def download_dataset():
    """Downloads the Telco Customer Churn dataset if not present locally."""
    os.makedirs('data', exist_ok=True)
    file_path = 'data/Telco-Customer-Churn.csv'
    
    if not os.path.exists(file_path):
        print("Downloading Telco Customer Churn dataset...")
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset found locally.")
        
    return file_path


def train_and_evaluate():
    file_path = download_dataset()
    df = pd.read_csv(file_path)

    # 1. Clean Data & Targets
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Selected practical features for app input
    feature_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'Contract', 'PaymentMethod', 'InternetService', 
        'TechSupport', 'OnlineSecurity', 'PaperlessBilling'
    ]
    
    X = df[feature_cols]
    y = df['Churn']

    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling']

    # 2. Robust Sub-Pipelines with Imputers to automatically fill any NaNs
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ]
    )

    # Train-test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model 1: Logistic Regression (Exp 2)
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    # 4. Model 2: Random Forest Ensemble (Exp 4)
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)

    # 5. Evaluate and Print Results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)

    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_acc = accuracy_score(y_test, lr_preds)

    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"\n1. LOGISTIC REGRESSION (Exp 2):")
    print(f"   Accuracy: {lr_acc * 100:.2f}%")
    print(f"   ROC-AUC:  {lr_auc:.4f}")

    print(f"\n2. RANDOM FOREST ENSEMBLE (Exp 4):")
    print(f"   Accuracy: {rf_acc * 100:.2f}%")
    print(f"   ROC-AUC:  {rf_auc:.4f}")

    print("\nDetailed Classification Report (Random Forest):")
    print(classification_report(y_test, rf_preds))

    # 6. Save Best Model Pipeline
    joblib.dump(rf_pipeline, 'churn_model.pkl')
    print("="*50)
    print(" Best pipeline saved successfully as 'churn_model.pkl'")
    print("="*50)


if __name__ == "__main__":
    train_and_evaluate()