"""
Load and save Diabetes dataset
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

def load_and_save_diabetes():
    """Load diabetes dataset and save as CSV"""
    print("📊 Loading Diabetes dataset...")
    
    # Load from sklearn
    diabetes = load_diabetes()
    
    # Create DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Save to CSV
    df.to_csv('data/diabetes.csv', index=False)
    
    print(f"✅ Saved diabetes.csv")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(diabetes.feature_names)}")
    print(f"   Feature names: {diabetes.feature_names}")
    print(f"\nTarget info:")
    print(f"   Min: {df['target'].min():.1f}")
    print(f"   Max: {df['target'].max():.1f}")
    print(f"   Mean: {df['target'].mean():.1f}")
    
    return df

if __name__ == "__main__":
    load_and_save_diabetes()
