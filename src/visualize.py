"""
Create model comparison visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")

def load_results():
    """Load comparison results"""
    with open('models/comparison_results.json', 'r') as f:
        return json.load(f)

def plot_model_comparison(results):
    """Create comprehensive comparison plots"""
    print("🎨 Creating visualizations...")
    
    # Prepare data
    models = [k for k in results.keys() if k != 'best_model']
    best_model = results['best_model']
    
    # Metrics to compare
    metrics = ['r2', 'rmse', 'mae']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Test R² Comparison
    ax1 = plt.subplot(2, 3, 1)
    r2_scores = [results[m]['test']['r2'] for m in models]
    colors = ['gold' if m == best_model else 'skyblue' for m in models]
    bars = ax1.bar(models, r2_scores, color=colors)
    ax1.set_title('Test R² Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. RMSE Comparison
    ax2 = plt.subplot(2, 3, 2)
    rmse_scores = [results[m]['test']['rmse'] for m in models]
    bars = ax2.bar(models, rmse_scores, color=colors)
    ax2.set_title('Test RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, rmse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom')
    
    # 3. Training Time
    ax3 = plt.subplot(2, 3, 3)
    times = [results[m]['training_time'] for m in models]
    bars = ax3.bar(models, times, color=colors)
    ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}s', ha='center', va='bottom')
    
    # 4. Train vs Val vs Test Performance
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(models))
    width = 0.25
    
    train_r2 = [results[m]['train']['r2'] for m in models]
    val_r2 = [results[m]['val']['r2'] for m in models]
    test_r2 = [results[m]['test']['r2'] for m in models]
    
    ax4.bar(x - width, train_r2, width, label='Train', alpha=0.8)
    ax4.bar(x, val_r2, width, label='Validation', alpha=0.8)
    ax4.bar(x + width, test_r2, width, label='Test', alpha=0.8)
    
    ax4.set_title('R² Across Datasets', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R² Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45)
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    # 5. MAE Comparison
    ax5 = plt.subplot(2, 3, 5)
    mae_scores = [results[m]['test']['mae'] for m in models]
    bars = ax5.bar(models, mae_scores, color=colors)
    ax5.set_title('Test MAE Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('MAE')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom')
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = []
    for m in models:
        marker = '🏆' if m == best_model else ''
        table_data.append([
            f"{m} {marker}",
            f"{results[m]['test']['r2']:.3f}",
            f"{results[m]['test']['rmse']:.1f}",
            f"{results[m]['training_time']:.2f}s"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'R²', 'RMSE', 'Time'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved plots/model_comparison.png")
    plt.close()

def main():
    """Main visualization"""
    print("="*60)
    print("📊 MODEL COMPARISON VISUALIZATION")
    print("="*60)
    
    results = load_results()
    plot_model_comparison(results)
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
