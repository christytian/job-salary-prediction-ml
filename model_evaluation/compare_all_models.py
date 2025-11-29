"""
Unified Model Comparison on Hybrid Features
============================================

This script compares all models (Linear, MLP, XGBoost, Random Forest) on the same
feature set (Hybrid: TF-IDF + Word2Vec) to determine which model performs best.

Key principle: Fair comparison requires same features, same train/test split, same metrics.
"""

import csv
import numpy as np
import sys
import os
from typing import Tuple, Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be skipped.")

try:
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Model comparison will be skipped.")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. XGBoost will be skipped.")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features(input_file: str) -> Tuple:
    """
    Load features and target from CSV file.
    
    Returns:
        Tuple of (X, y, feature_info)
    """
    print("=" * 80)
    print("UNIFIED MODEL COMPARISON")
    print("=" * 80)
    print(f"\n📖 Loading data from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"   Read {len(data):,} records")
    print(f"   Total columns: {len(fieldnames) if fieldnames else 0}")
    
    # Extract target variable
    target_col = 'salary_normalized'
    if target_col not in fieldnames:
        print(f"\n❌ Error: Target column '{target_col}' not found.")
        return None, None, None
    
    # Get all feature columns (exclude target and non-feature columns)
    excluded_cols = ['salary_normalized', 'normalized_salary', 'benefits_list_missing']
    
    feature_cols = []
    for col in fieldnames:
        if col in excluded_cols:
            continue
        feature_cols.append(col)
    
    # Count feature types
    tfidf_cols = [c for c in feature_cols if '_tfidf_' in c]
    w2v_cols = [c for c in feature_cols if '_w2v_' in c]
    onehot_cols = [c for c in feature_cols if any(c.startswith(f'{feat}_') for feat in ['formatted_work_type', 'formatted_experience_level', 'company_size'])]
    frequency_cols = [c for c in feature_cols if '_frequency' in c]
    binary_cols = [c for c in feature_cols if '_binary' in c]
    other_numeric_cols = [c for c in feature_cols if c not in tfidf_cols + w2v_cols + onehot_cols + frequency_cols + binary_cols]
    
    print("\n📊 Feature breakdown:")
    print(f"   - TF-IDF: {len(tfidf_cols)}")
    print(f"   - Word2Vec: {len(w2v_cols)}")
    print(f"   - One-Hot: {len(onehot_cols)}")
    print(f"   - Frequency: {len(frequency_cols)}")
    print(f"   - Binary: {len(binary_cols)}")
    print(f"   - Other numeric: {len(other_numeric_cols)}")
    print(f"   - Total features: {len(feature_cols)}")
    
    # Prepare data
    X_list = []
    y_list = []
    
    for row in data:
        # Extract target
        try:
            target = float(row.get(target_col, 0))
            if target <= 0:
                continue
        except (ValueError, TypeError):
            continue
        
        # Extract features
        features = []
        for col in feature_cols:
            try:
                val = float(row.get(col, 0))
                features.append(val)
            except (ValueError, TypeError):
                features.append(0.0)
        
        if len(features) == len(feature_cols):
            X_list.append(features)
            y_list.append(target)
    
    if len(X_list) < 100:
        print(f"\n❌ Error: Only {len(X_list)} valid samples. Need at least 100.")
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✅ Valid samples: {len(X):,}")
    
    feature_info = {
        'total_features': len(feature_cols),
        'tfidf_count': len(tfidf_cols),
        'w2v_count': len(w2v_cols),
        'onehot_count': len(onehot_cols),
        'frequency_count': len(frequency_cols),
        'binary_count': len(binary_cols),
        'other_numeric_count': len(other_numeric_cols)
    }
    
    return X, y, feature_info


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_model(name: str, model, X_train, X_test, y_train, y_test, 
                             needs_scaling: bool = False, scaler: StandardScaler = None):
    """
    Train a model and evaluate its performance.
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n🤖 Training {name}...")
    
    # Scale if needed
    if needs_scaling and scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train model
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Check for overfitting
    overfitting_gap = train_r2 - test_r2
    
    # Assess overfitting severity
    if overfitting_gap < 0.05:
        overfitting_status = "None"
    elif overfitting_gap < 0.1:
        overfitting_status = "Mild"
    elif overfitting_gap < 0.2:
        overfitting_status = "Moderate"
    else:
        overfitting_status = "Severe"
    
    results = {
        'name': name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting_gap': overfitting_gap,
        'overfitting_status': overfitting_status,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Test RMSE: ${test_rmse:,.0f}")
    print(f"   Overfitting gap: {overfitting_gap:.4f}")
    
    return results


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_all_models(input_file: str):
    """
    Compare all models on the same feature set.
    """
    if not SKLEARN_AVAILABLE:
        print("\n❌ Error: sklearn not available. Cannot run comparison.")
        return
    
    # Load data
    X, y, feature_info = load_features(input_file)
    if X is None:
        return
    
    # Same train/test split for ALL models (fair comparison)
    print("\n📊 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Prepare scaler (for models that need scaling)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Initialize all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    all_results = []
    
    # 1. Linear Regression (Ridge)
    if SKLEARN_AVAILABLE:
        ridge_model = Ridge(alpha=1.0, max_iter=1000)
        results = train_and_evaluate_model(
            "Ridge Regression", ridge_model, 
            X_train, X_test, y_train, y_test,
            needs_scaling=True, scaler=scaler
        )
        if results:
            all_results.append(results)
    
    # 2. MLP Neural Network
    if SKLEARN_AVAILABLE:
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        results = train_and_evaluate_model(
            "MLP Neural Network", mlp_model,
            X_train, X_test, y_train, y_test,
            needs_scaling=True, scaler=scaler
        )
        if results:
            all_results.append(results)
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        results = train_and_evaluate_model(
            "XGBoost", xgb_model,
            X_train, X_test, y_train, y_test,
            needs_scaling=False  # XGBoost doesn't need scaling
        )
        if results:
            all_results.append(results)
    
    # 4. Random Forest
    if SKLEARN_AVAILABLE:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        results = train_and_evaluate_model(
            "Random Forest", rf_model,
            X_train, X_test, y_train, y_test,
            needs_scaling=False  # Random Forest doesn't need scaling
        )
        if results:
            all_results.append(results)
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    
    if not all_results:
        print("\n❌ No models were successfully trained.")
        return
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Sort by Test R² (descending)
    all_results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    print(f"\n{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Overfit Gap':>12} {'Overfit Status':>15}")
    print("=" * 95)
    
    for r in all_results:
        status_icon = "✅" if r['overfitting_status'] == "None" else "⚠️" if r['overfitting_status'] == "Mild" else "🔴"
        print(f"{r['name']:<20} {r['train_r2']:>10.4f} {r['test_r2']:>10.4f} {r['overfitting_gap']:>12.4f} {status_icon} {r['overfitting_status']:>10}")
    
    # Winner
    winner = all_results[0]
    print("\n" + "=" * 80)
    print("🏆 WINNER")
    print("=" * 80)
    print(f"\nBest Model: {winner['name']}")
    print(f"   Train R²:  {winner['train_r2']:.4f}")
    print(f"   Test R²:   {winner['test_r2']:.4f}")
    print(f"   Test RMSE: ${winner['test_rmse']:,.0f}")
    print(f"   Test MAE:  ${winner['test_mae']:,.0f}")
    print(f"   Overfitting gap: {winner['overfitting_gap']:.4f}")
    print(f"   Overfitting status: {winner['overfitting_status']}")
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for i, r in enumerate(all_results, 1):
        status_icon = "✅" if r['overfitting_status'] == "None" else "⚠️" if r['overfitting_status'] == "Mild" else "🔴"
        print(f"\n{i}. {r['name']} {status_icon} Overfitting: {r['overfitting_status']}")
        print(f"   Train R²:  {r['train_r2']:.4f}")
        print(f"   Test R²:   {r['test_r2']:.4f}")
        print(f"   Train RMSE: ${r['train_rmse']:,.0f}")
        print(f"   Test RMSE:  ${r['test_rmse']:,.0f}")
        print(f"   Train MAE:  ${r['train_mae']:,.0f}")
        print(f"   Test MAE:   ${r['test_mae']:,.0f}")
        print(f"   Overfitting gap: {r['overfitting_gap']:.4f} (Train R² - Test R², lower is better)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Compared {len(all_results)} models on same feature set")
    print("✅ Used same train/test split (random_state=42)")
    print(f"✅ Winner: {winner['name']} with Test R² = {winner['test_r2']:.4f}")
    print(f"   Train R² = {winner['train_r2']:.4f}, Overfitting status: {winner['overfitting_status']}")
    
    # Overfitting assessment
    if winner['overfitting_status'] == "Severe":
        print(f"\n🔴 Warning: Severe overfitting detected (gap = {winner['overfitting_gap']:.4f})")
        print("   Strongly recommend: regularization, simpler model, or more data")
    elif winner['overfitting_status'] == "Moderate":
        print(f"\n⚠️  Warning: Moderate overfitting detected (gap = {winner['overfitting_gap']:.4f})")
        print("   Consider: regularization, early stopping, or feature selection")
    elif winner['overfitting_status'] == "Mild":
        print(f"\n⚠️  Note: Mild overfitting detected (gap = {winner['overfitting_gap']:.4f})")
        print("   Model is acceptable but could be improved with slight regularization")
    else:
        print(f"\n✅ Good: No significant overfitting (gap = {winner['overfitting_gap']:.4f})")
    
    print("\n" + "=" * 80)
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    if MATPLOTLIB_AVAILABLE:
        print("\n📊 Generating visualizations...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..')
        
        # Plot 1: Predicted vs Actual (Test set)
        plot_predicted_vs_actual(all_results, output_dir)
        
        # Plot 2: Overfitting Visualization (Train R² vs Test R²)
        plot_overfitting(all_results, output_dir)
        
        print("   ✅ Visualizations saved to project root directory")
    else:
        print("\n⚠️  matplotlib not available. Skipping visualizations.")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_predicted_vs_actual(all_results: List[Dict], output_dir: str):
    """
    Plot predicted vs actual values for all models (Test set).
    Each model gets its own subplot.
    """
    n_models = len(all_results)
    if n_models == 0:
        return
    
    # Determine subplot layout
    if n_models <= 2:
        nrows, ncols = 1, n_models
    elif n_models <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    
    _, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        
        y_test = result['y_test']
        y_test_pred = result['y_test_pred']
        model_name = result['name']
        test_r2 = result['test_r2']
        
        # Scatter plot
        ax.scatter(y_test, y_test_pred, alpha=0.5, s=20, edgecolors='none')
        
        # Perfect prediction line (y=x)
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual Salary ($)', fontsize=11)
        ax.set_ylabel('Predicted Salary ($)', fontsize=11)
        ax.set_title(f'{model_name}\nTest R² = {test_r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_predicted_vs_actual.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("   📈 Saved: predicted_vs_actual.png")


def plot_overfitting(all_results: List[Dict], output_dir: str):
    """
    Plot Train R² vs Test R² to visualize overfitting.
    """
    if len(all_results) == 0:
        return
    
    _, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    model_names = [r['name'] for r in all_results]
    train_r2 = [r['train_r2'] for r in all_results]
    test_r2 = [r['test_r2'] for r in all_results]
    overfitting_gaps = [r['overfitting_gap'] for r in all_results]
    
    # Color mapping based on overfitting status
    colors = []
    for r in all_results:
        status = r['overfitting_status']
        if status == "None":
            colors.append('green')
        elif status == "Mild":
            colors.append('orange')
        elif status == "Moderate":
            colors.append('red')
        else:  # Severe
            colors.append('darkred')
    
    # Scatter plot
    ax.scatter(train_r2, test_r2, c=colors, s=150, alpha=0.7, 
               edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add model labels
    for i, name in enumerate(model_names):
        ax.annotate(name, (train_r2[i], test_r2[i]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    # Perfect model line (no overfitting: Train R² = Test R²)
    all_r2 = train_r2 + test_r2
    min_r2 = min(all_r2) - 0.05
    max_r2 = max(all_r2) + 0.05
    ax.plot([min_r2, max_r2], [min_r2, max_r2], 
            'b--', linewidth=2, label='No Overfitting (Train R² = Test R²)', zorder=1)
    
    # Labels and title
    ax.set_xlabel('Train R²', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax.set_title('Model Overfitting Visualization\n(Train R² vs Test R²)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend for overfitting status
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='No Overfitting (gap < 0.05)'),
        Patch(facecolor='orange', edgecolor='black', label='Mild Overfitting (0.05 ≤ gap < 0.1)'),
        Patch(facecolor='red', edgecolor='black', label='Moderate Overfitting (0.1 ≤ gap < 0.2)'),
        Patch(facecolor='darkred', edgecolor='black', label='Severe Overfitting (gap ≥ 0.2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    # Add text annotations for gaps
    for i, (name, gap) in enumerate(zip(model_names, overfitting_gaps)):
        ax.text(train_r2[i], test_r2[i] - 0.02, f'gap={gap:.3f}', 
               fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_overfitting.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("   📈 Saved: overfitting_visualization.png")


def main():
    """Main entry point."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    # Default input file (Hybrid features with categorical encoding)
    input_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_onehot_features.csv')
    
    # Allow command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    try:
        compare_all_models(input_file)
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {input_file}")
        print("   Please run hybrid.py and onehot_encode.py first to generate features.")
        sys.exit(1)
    except (OSError, KeyError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

