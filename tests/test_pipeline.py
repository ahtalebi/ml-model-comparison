"""
Tests for ML pipeline
"""

import unittest
import os
import json
import joblib
import pandas as pd

class TestMLPipeline(unittest.TestCase):
    
    def test_1_data_exists(self):
        """Test data file exists"""
        self.assertTrue(os.path.exists('data/diabetes.csv'))
    
    def test_2_data_valid(self):
        """Test data is valid"""
        df = pd.read_csv('data/diabetes.csv')
        self.assertEqual(len(df.columns), 11)  # 10 features + target
        self.assertGreater(len(df), 400)
    
    def test_3_models_exist(self):
        """Test all models were created"""
        models = ['linear_regression', 'ridge', 'random_forest', 'xgboost', 'best_model']
        for model in models:
            self.assertTrue(os.path.exists(f'models/{model}.pkl'))
    
    def test_4_results_exist(self):
        """Test results file exists"""
        self.assertTrue(os.path.exists('models/comparison_results.json'))
    
    def test_5_best_model_selected(self):
        """Test best model was selected"""
        with open('models/comparison_results.json') as f:
            results = json.load(f)
        self.assertIn('best_model', results)
        self.assertIn(results['best_model'], ['linear_regression', 'ridge', 'random_forest', 'xgboost'])
    
    def test_6_model_performance(self):
        """Test models meet minimum performance"""
        with open('models/comparison_results.json') as f:
            results = json.load(f)
        
        for model in ['linear_regression', 'ridge', 'random_forest', 'xgboost']:
            r2 = results[model]['test']['r2']
            self.assertGreater(r2, 0.3, f"{model} R² too low: {r2}")
    
    def test_7_plots_created(self):
        """Test plots were generated"""
        self.assertTrue(os.path.exists('plots/model_comparison.png'))

if __name__ == "__main__":
    unittest.main(verbosity=2)
