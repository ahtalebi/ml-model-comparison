"""
Train and compare multiple models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time
import os

def load_data():
    """Load the diabetes dataset"""
    print("📊 Loading data...")
    df = pd.read_csv('data/diabetes.csv')
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def split_data(X, y):
    """Split into train/val/test"""
    print("\n✂️ Splitting data...")
    
    # 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_single_model(name, model, X_train, y_train):
    """Train a single model and return training time"""
    print(f"\n🚀 Training {name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"  ✅ Trained in {training_time:.2f}s")
    return model, training_time

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    
    metrics = {
        'r2': float(r2_score(y, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred))
    }
    
    print(f"\n  📊 {dataset_name} Metrics:")
    print(f"     R²:   {metrics['r2']:.4f}")
    print(f"     RMSE: {metrics['rmse']:.2f}")
    print(f"     MAE:  {metrics['mae']:.2f}")
    
    return metrics

def train_all_models():
    """Train all models and compare"""
    print("="*60)
    print("🎯 MULTI-MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load and split data
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Define models
    models_config = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    # Train all models
    results = {}
    trained_models = {}
    
    for name, model in models_config.items():
        # Train
        trained_model, train_time = train_single_model(name, model, X_train, y_train)
        
        # Evaluate
        train_metrics = evaluate_model(trained_model, X_train, y_train, "TRAIN")
        val_metrics = evaluate_model(trained_model, X_val, y_val, "VAL")
        test_metrics = evaluate_model(trained_model, X_test, y_test, "TEST")
        
        # Store results
        results[name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'training_time': train_time
        }
        
        trained_models[name] = trained_model
        
        # Save individual model
        os.makedirs('models', exist_ok=True)
        joblib.dump(trained_model, f'models/{name}.pkl')
        print(f"  💾 Saved models/{name}.pkl")
    
    # Select best model (based on validation R²)
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['val']['r2'])
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   Validation R²: {results[best_model_name]['val']['r2']:.4f}")
    print("="*60)
    
    # Save best model
    joblib.dump(trained_models[best_model_name], 'models/best_model.pkl')
    print(f"💾 Saved models/best_model.pkl")
    
    # Save all results
    results['best_model'] = best_model_name
    with open('models/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved models/comparison_results.json")
    
    return results, best_model_name

if __name__ == "__main__":
    train_all_models()
