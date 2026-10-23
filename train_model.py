import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def generate_food_delivery_data(samples=5000):
    """Generates a realistic Zomato/Swiggy delivery dataset."""
    np.random.seed(42)
    
    distance_km = np.random.uniform(1.0, 20.0, samples)
    prep_time_min = np.random.uniform(5.0, 35.0, samples)
    courier_exp_yrs = np.random.uniform(0.5, 8.0, samples)
    
    weather_opts = ['Sunny', 'Rainy', 'Foggy', 'Stormy']
    weather = np.random.choice(weather_opts, samples, p=[0.5, 0.3, 0.15, 0.05])
    
    traffic_opts = ['Low', 'Medium', 'High', 'Jam']
    traffic = np.random.choice(traffic_opts, samples, p=[0.25, 0.45, 0.2, 0.1])
    
    vehicle_opts = ['Scooter', 'Motorcycle', 'Electric Bike']
    vehicle = np.random.choice(vehicle_opts, samples, p=[0.5, 0.4, 0.1])
    
    # Calculate delivery time formula with realistic physical factors
    base_time = 8.0
    transit_speed = np.where(vehicle == 'Electric Bike', 2.8, 2.3) # mins per km
    
    weather_delay = np.where(weather == 'Rainy', 7, np.where(weather == 'Foggy', 5, np.where(weather == 'Stormy', 14, 0)))
    traffic_delay = np.where(traffic == 'Medium', 4, np.where(traffic == 'High', 10, np.where(traffic == 'Jam', 18, 0)))
    exp_discount = courier_exp_yrs * 0.6
    
    noise = np.random.normal(0, 3, samples)
    
    delivery_time_min = (
        base_time + 
        (distance_km * transit_speed) + 
        prep_time_min + 
        weather_delay + 
        traffic_delay - 
        exp_discount + 
        noise
    )
    
    df = pd.DataFrame({
        'distance_km': np.round(distance_km, 2),
        'prep_time_min': np.round(prep_time_min, 1),
        'courier_exp_yrs': np.round(courier_exp_yrs, 1),
        'weather': weather,
        'traffic_level': traffic,
        'vehicle_type': vehicle,
        'delivery_time_min': np.round(np.clip(delivery_time_min, 12, 120), 1)
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/zomato_delivery_data.csv', index=False)
    return df


def train_and_evaluate():
    df = generate_food_delivery_data()

    X = df.drop(columns=['delivery_time_min'])
    y = df['delivery_time_min']

    num_features = ['distance_km', 'prep_time_min', 'courier_exp_yrs']
    cat_features = ['weather', 'traffic_level', 'vehicle_type']

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Multivariate Linear Regression (Exp 1)
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    # Model 2: Random Forest Regressor Ensemble (Exp 4)
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)

    # Evaluation
    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    print("\n" + "="*50)
    print("FOOD DELIVERY TIME MODEL EVALUATION")
    print("="*50)
    print(f"\n1. MULTIVARIATE LINEAR REGRESSION (Exp 1):")
    print(f"   Mean Absolute Error (MAE): {mean_absolute_error(y_test, lr_preds):.2f} mins")
    print(f"   R2 Score:                  {r2_score(y_test, lr_preds):.4f}")

    print(f"\n2. RANDOM FOREST ENSEMBLE REGRESSOR (Exp 4):")
    print(f"   Mean Absolute Error (MAE): {mean_absolute_error(y_test, rf_preds):.2f} mins")
    print(f"   R2 Score:                  {r2_score(y_test, rf_preds):.4f}")

    joblib.dump(rf_pipeline, 'delivery_model.pkl')
    print("="*50)
    print(" Production model saved successfully as 'delivery_model.pkl'")
    print("="*50)


if __name__ == "__main__":
    train_and_evaluate()