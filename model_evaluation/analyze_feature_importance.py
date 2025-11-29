"""
Feature Importance Analysis
===========================

This script analyzes feature importance across all models (Ridge, MLP, XGBoost, Random Forest)
to identify which features are most important for salary prediction.

Key features:
- Calculates feature importance using model-specific methods
- Analyzes importance by feature type (TF-IDF, Word2Vec, One-Hot, etc.)
- Visualizes top features and feature type importance
- Provides recommendations for feature engineering
"""

import csv
import numpy as np
import sys
import os
from typing import Tuple, Dict, List
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass  # seaborn is optional, continue without it
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
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Feature importance analysis will be skipped.")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Don't print warning here, will print later if needed


# =============================================================================
# DATA LOADING (Reused from compare_all_models.py)
# =============================================================================

def load_features(input_file: str) -> Tuple:
    """
    Load features and target from CSV file.
    
    Returns:
        Tuple of (X, y, feature_cols, feature_info)
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print(f"\n📖 Loading data from: {input_file}")
    
    # Read CSV file with progress indication
    print("   Reading CSV file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = []
        row_count = 0
        for row in reader:
            data.append(row)
            row_count += 1
            if row_count % 5000 == 0:
                print(f"   ... read {row_count:,} rows", end='\r')
    
    print(f"\n   Read {len(data):,} records")
    print(f"   Total columns: {len(fieldnames) if fieldnames else 0}")
    
    # Extract target variable
    target_col = 'salary_normalized'
    if target_col not in fieldnames:
        print(f"\n❌ Error: Target column '{target_col}' not found.")
        return None, None, None, None
    
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
    
    # Prepare data with progress indication
    print("\n   Processing data...")
    X_list = []
    y_list = []
    
    total_rows = len(data)
    processed = 0
    
    for row in data:
        # Extract target
        try:
            target = float(row.get(target_col, 0))
            if target <= 0:
                processed += 1
                if processed % 5000 == 0:
                    print(f"   ... processed {processed:,}/{total_rows:,} rows", end='\r')
                continue
        except (ValueError, TypeError):
            processed += 1
            if processed % 5000 == 0:
                print(f"   ... processed {processed:,}/{total_rows:,} rows", end='\r')
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
        
        processed += 1
        if processed % 5000 == 0:
            print(f"   ... processed {processed:,}/{total_rows:,} rows (valid: {len(X_list):,})", end='\r')
    
    print(f"\n   Processed {processed:,} rows")
    
    if len(X_list) < 100:
        print(f"\n❌ Error: Only {len(X_list)} valid samples. Need at least 100.")
        return None, None, None, None
    
    print("   Converting to numpy arrays...")
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
        'other_numeric_count': len(other_numeric_cols),
        'tfidf_cols': tfidf_cols,
        'w2v_cols': w2v_cols,
        'onehot_cols': onehot_cols,
        'frequency_cols': frequency_cols,
        'binary_cols': binary_cols,
        'other_numeric_cols': other_numeric_cols
    }
    
    return X, y, feature_cols, feature_info


# =============================================================================
# FEATURE TYPE CLASSIFICATION
# =============================================================================

def classify_feature_type(feature_name: str, feature_info: Dict) -> str:
    """Classify a feature into its type."""
    if feature_name in feature_info['tfidf_cols']:
        return 'TF-IDF'
    elif feature_name in feature_info['w2v_cols']:
        return 'Word2Vec'
    elif feature_name in feature_info['onehot_cols']:
        return 'One-Hot'
    elif feature_name in feature_info['frequency_cols']:
        return 'Frequency'
    elif feature_name in feature_info['binary_cols']:
        return 'Binary'
    elif feature_name in feature_info['other_numeric_cols']:
        return 'Other Numeric'
    else:
        return 'Unknown'


# =============================================================================
# FEATURE IMPORTANCE CALCULATION
# =============================================================================

def calculate_ridge_importance(model, feature_cols: List[str]) -> Dict:
    """Calculate feature importance for Ridge Regression using coefficient magnitude."""
    coef_abs = np.abs(model.coef_)
    # Normalize to sum to 1
    coef_normalized = coef_abs / coef_abs.sum() if coef_abs.sum() > 0 else coef_abs
    
    importance_dict = {}
    for idx, feature_name in enumerate(feature_cols):
        importance_dict[feature_name] = coef_normalized[idx]
    
    return importance_dict


def calculate_mlp_importance(model, X_test, y_test, feature_cols: List[str], 
                            n_repeats: int = 5) -> Dict:
    """Calculate feature importance for MLP using permutation importance."""
    print("   Computing permutation importance (this may take a while)...")
    
    # Use a smaller sample for faster computation
    sample_size = min(500, len(X_test))
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    perm_result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='r2'
    )
    
    importance_dict = {}
    importances = perm_result.importances_mean
    # Normalize to sum to 1
    importances_normalized = importances / importances.sum() if importances.sum() > 0 else importances
    
    for idx, feature_name in enumerate(feature_cols):
        importance_dict[feature_name] = importances_normalized[idx]
    
    return importance_dict


def calculate_tree_importance(model, feature_cols: List[str]) -> Dict:
    """Calculate feature importance for tree-based models (XGBoost, Random Forest)."""
    importances = model.feature_importances_
    # Normalize to sum to 1
    importances_normalized = importances / importances.sum() if importances.sum() > 0 else importances
    
    importance_dict = {}
    for idx, feature_name in enumerate(feature_cols):
        importance_dict[feature_name] = importances_normalized[idx]
    
    return importance_dict


# =============================================================================
# MODEL TRAINING AND IMPORTANCE EXTRACTION
# =============================================================================

def train_and_extract_importance(name: str, model, X_train, X_test, y_train, y_test,
                                 feature_cols: List[str], feature_info: Dict,
                                 needs_scaling: bool = False, scaler: StandardScaler = None) -> Dict:
    """Train a model and extract feature importance."""
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
        print("   ✅ Training completed successfully")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate feature importance based on model type
    try:
        if name == "Ridge Regression":
            importance_dict = calculate_ridge_importance(model, feature_cols)
            print("   ✅ Calculated Ridge importance using coefficients")
        elif name == "MLP Neural Network":
            importance_dict = calculate_mlp_importance(model, X_test_scaled, y_test, feature_cols)
            print("   ✅ Calculated MLP importance using permutation importance")
        elif name in ["XGBoost", "Random Forest"]:
            importance_dict = calculate_tree_importance(model, feature_cols)
            print(f"   ✅ Calculated {name} importance using feature_importances_")
        else:
            print("   ⚠️  Unknown model type, skipping importance calculation")
            return None
    except Exception as e:
        print(f"   ❌ Error calculating importance: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate performance metrics
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"   Test R²: {test_r2:.4f}, Test RMSE: ${test_rmse:,.0f}")
    
    # Aggregate importance by feature type
    type_importance = defaultdict(float)
    for feature_name, importance in importance_dict.items():
        feature_type = classify_feature_type(feature_name, feature_info)
        type_importance[feature_type] += importance
    
    result = {
        'name': name,
        'importance_dict': importance_dict,
        'type_importance': dict(type_importance),
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }
    
    return result


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_top_features(all_results: List[Dict], feature_cols: List[str], 
                        feature_info: Dict, top_n: int = 30) -> Dict:
    """Analyze top features across all models."""
    print("\n" + "=" * 80)
    print("TOP FEATURES ANALYSIS")
    print("=" * 80)
    
    # Aggregate importance across all models
    aggregated_importance = defaultdict(float)
    model_count = defaultdict(int)
    
    for result in all_results:
        for feature_name, importance in result['importance_dict'].items():
            aggregated_importance[feature_name] += importance
            model_count[feature_name] += 1
    
    # Average importance across models
    avg_importance = {k: v / model_count[k] for k, v in aggregated_importance.items()}
    
    # Sort by average importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 Top {top_n} Features (Average Importance Across All Models):")
    print(f"\n{'Rank':<6} {'Feature Name':<60} {'Avg Importance':>15} {'Type':<15} {'Models':<10}")
    print("=" * 110)
    
    top_features = []
    for rank, (feature_name, importance) in enumerate(sorted_features[:top_n], 1):
        feature_type = classify_feature_type(feature_name, feature_info)
        models_using = model_count[feature_name]
        top_features.append({
            'rank': rank,
            'name': feature_name,
            'importance': importance,
            'type': feature_type,
            'models_count': models_using
        })
        print(f"{rank:<6} {feature_name:<60} {importance:>15.6f} {feature_type:<15} {models_using:<10}")
    
    return {
        'top_features': top_features,
        'avg_importance': avg_importance,
        'model_count': dict(model_count)
    }


def analyze_feature_type_importance(all_results: List[Dict]) -> Dict:
    """Analyze importance aggregated by feature type."""
    print("\n" + "=" * 80)
    print("FEATURE TYPE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Aggregate by type across all models
    type_importance_all = defaultdict(lambda: defaultdict(float))
    
    for result in all_results:
        model_name = result['name']
        for feature_type, importance in result['type_importance'].items():
            type_importance_all[feature_type][model_name] = importance
    
    # Calculate average across models
    type_avg_importance = {}
    for feature_type, model_importances in type_importance_all.items():
        avg = np.mean(list(model_importances.values()))
        type_avg_importance[feature_type] = avg
    
    # Sort by average importance
    sorted_types = sorted(type_avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Feature Type Importance (Average Across All Models):")
    print(f"\n{'Rank':<6} {'Feature Type':<20} {'Avg Importance':>15} {'Percentage':>12}")
    print("=" * 60)
    
    total_importance = sum(type_avg_importance.values())
    for rank, (feature_type, importance) in enumerate(sorted_types, 1):
        percentage = (importance / total_importance * 100) if total_importance > 0 else 0
        print(f"{rank:<6} {feature_type:<20} {importance:>15.6f} {percentage:>11.2f}%")
    
    return {
        'type_importance_all': dict(type_importance_all),
        'type_avg_importance': type_avg_importance,
        'sorted_types': sorted_types
    }


def find_consistent_features(all_results: List[Dict], feature_cols: List[str],
                             top_n_per_model: int = 20) -> Dict:
    """Find features that are important across multiple models."""
    print("\n" + "=" * 80)
    print("CROSS-MODEL CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # Get top features for each model
    model_top_features = {}
    for result in all_results:
        model_name = result['name']
        sorted_features = sorted(
            result['importance_dict'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:top_n_per_model]]
        model_top_features[model_name] = set(top_features)
    
    # Count how many models consider each feature as top
    feature_model_count = defaultdict(int)
    for model_name, top_features in model_top_features.items():
        for feature in top_features:
            feature_model_count[feature] += 1
    
    # Find features that appear in multiple models
    consistent_features = {}
    for feature, count in feature_model_count.items():
        if count >= 2:  # Appears in at least 2 models
            consistent_features[feature] = count
    
    sorted_consistent = sorted(consistent_features.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ Features appearing in top {top_n_per_model} of multiple models:")
    print(f"\n{'Feature Name':<60} {'Models Count':>15}")
    print("=" * 80)
    
    for feature, count in sorted_consistent[:30]:  # Show top 30
        print(f"{feature:<60} {count:>15}")
    
    return {
        'consistent_features': dict(consistent_features),
        'model_top_features': {k: list(v) for k, v in model_top_features.items()}
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_top_features_by_model(all_results: List[Dict], feature_info: Dict, 
                               output_dir: str, top_n: int = 20):
    """Plot top features for each model."""
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
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color map for feature types
    type_colors = {
        'TF-IDF': '#1f77b4',
        'Word2Vec': '#ff7f0e',
        'One-Hot': '#2ca02c',
        'Frequency': '#d62728',
        'Binary': '#9467bd',
        'Other Numeric': '#8c564b',
        'Unknown': '#7f7f7f'
    }
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        model_name = result['name']
        importance_dict = result['importance_dict']
        
        # Get top N features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Extract data
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        feature_types = [classify_feature_type(name, feature_info) for name in feature_names]
        colors = [type_colors.get(ft, '#7f7f7f') for ft in feature_types]
        
        # Truncate long feature names
        display_names = [name[:50] + '...' if len(name) > 50 else name for name in feature_names]
        
        # Horizontal bar plot
        y_pos = np.arange(len(display_names))
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\nTop {top_n} Features', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Top feature at top
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=ftype)
                      for ftype, color in type_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(type_colors),
              bbox_to_anchor=(0.5, -0.02), fontsize=9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, 'feature_importance_top_features_by_model.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved: feature_importance_top_features_by_model.png")


def plot_feature_type_importance(all_results: List[Dict], output_dir: str):
    """Plot feature type importance comparison across models."""
    if len(all_results) == 0:
        return
    
    # Prepare data
    feature_types = set()
    for result in all_results:
        feature_types.update(result['type_importance'].keys())
    feature_types = sorted(list(feature_types))
    
    model_names = [r['name'] for r in all_results]
    data_matrix = []
    
    for model_name in model_names:
        row = []
        for ftype in feature_types:
            # Find the result for this model
            result = next((r for r in all_results if r['name'] == model_name), None)
            if result:
                importance = result['type_importance'].get(ftype, 0)
                row.append(importance)
            else:
                row.append(0)
        data_matrix.append(row)
    
    # Create heatmap
    _, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_types)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(feature_types, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(feature_types)):
            ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                   ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Feature Type Importance Across Models', fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Importance')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance_type_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved: feature_importance_type_heatmap.png")
    
    # Also create a grouped bar chart
    _, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(feature_types))
    width = 0.2
    multiplier = 0
    
    for model_name in model_names:
        result = next((r for r in all_results if r['name'] == model_name), None)
        if result:
            importances = [result['type_importance'].get(ftype, 0) for ftype in feature_types]
            offset = width * multiplier
            ax.bar(x + offset, importances, width, label=model_name, alpha=0.8)
            multiplier += 1
    
    ax.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Type Importance Comparison Across Models', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(feature_types, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance_type_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved: feature_importance_type_comparison.png")


def plot_aggregated_top_features(analysis_result: Dict, feature_info: Dict,
                                 output_dir: str, top_n: int = 30):
    """Plot aggregated top features across all models."""
    top_features = analysis_result['top_features'][:top_n]
    
    if len(top_features) == 0:
        return
    
    _, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.3)))
    
    # Extract data
    feature_names = [f['name'] for f in top_features]
    importances = [f['importance'] for f in top_features]
    feature_types = [f['type'] for f in top_features]
    models_count = [f['models_count'] for f in top_features]
    
    # Color map
    type_colors = {
        'TF-IDF': '#1f77b4',
        'Word2Vec': '#ff7f0e',
        'One-Hot': '#2ca02c',
        'Frequency': '#d62728',
        'Binary': '#9467bd',
        'Other Numeric': '#8c564b',
        'Unknown': '#7f7f7f'
    }
    colors = [type_colors.get(ft, '#7f7f7f') for ft in feature_types]
    
    # Truncate long names
    display_names = [name[:60] + '...' if len(name) > 60 else name for name in feature_names]
    
    # Horizontal bar plot
    y_pos = np.arange(len(display_names))
    ax.barh(y_pos, importances, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add model count annotations
    for i, (importance, count) in enumerate(zip(importances, models_count)):
        ax.text(importance * 1.02, i, f'({count} models)', 
               va='center', fontsize=8, style='italic')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel('Average Feature Importance (Across All Models)', 
                 fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Features (Aggregated Across All Models)', 
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=ftype)
                      for ftype, color in type_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance_aggregated_top.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved: feature_importance_aggregated_top.png")


def plot_all_models_top100(all_results: List[Dict], feature_info: Dict,
                           output_dir: str, top_n: int = 100):
    """Plot top 100 features for all models in one figure with subplots.
    Each model shows its own top N features."""
    if len(all_results) == 0:
        return
    
    n_models = len(all_results)
    
    # Determine subplot layout
    if n_models <= 2:
        nrows, ncols = 1, n_models
    elif n_models <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    
    # Color map for feature types
    type_colors = {
        'TF-IDF': '#1f77b4',
        'Word2Vec': '#ff7f0e',
        'One-Hot': '#2ca02c',
        'Frequency': '#d62728',
        'Binary': '#9467bd',
        'Other Numeric': '#8c564b',
        'Unknown': '#7f7f7f'
    }
    
    # Calculate figure height based on number of features
    fig_height = max(8, top_n * 0.06)
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, fig_height))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        model_name = result['name']
        importance_dict = result['importance_dict']
        
        # Get top N features for this model
        sorted_features = sorted(importance_dict.items(), 
                                key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Extract data
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        feature_types = [classify_feature_type(name, feature_info) for name in feature_names]
        colors = [type_colors.get(ft, '#7f7f7f') for ft in feature_types]
        
        # Truncate long feature names
        display_names = [name[:55] + '...' if len(name) > 55 else name for name in feature_names]
        
        # Horizontal bar plot
        y_pos = np.arange(len(display_names))
        ax.barh(y_pos, importances, color=colors, alpha=0.7, 
                edgecolor='black', linewidth=0.3)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=6)
        ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\nTop {top_n} Features', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=ftype)
                      for ftype, color in type_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(type_colors),
              bbox_to_anchor=(0.5, -0.01), fontsize=8)
    
    plt.suptitle(f'Top {top_n} Features Comparison Across All Models',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    output_path = os.path.join(output_dir, f'feature_importance_all_models_top{top_n}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved: feature_importance_all_models_top{top_n}.png")


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def generate_recommendations(all_results: List[Dict], analysis_result: Dict,
                            type_analysis: Dict, consistent_features: Dict,
                            feature_info: Dict) -> None:
    """Generate recommendations based on feature importance analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # 1. Most important feature types
    sorted_types = type_analysis['sorted_types']
    if sorted_types:
        top_type = sorted_types[0]
        print(f"\n1. 🏆 Most Important Feature Type: {top_type[0]}")
        print(f"   - Average importance: {top_type[1]:.4f}")
        print(f"   - Recommendation: Focus feature engineering efforts on {top_type[0]} features")
    
    # 2. Feature types to consider removing
    if len(sorted_types) > 1:
        bottom_type = sorted_types[-1]
        if bottom_type[1] < 0.01:  # Less than 1% importance
            print(f"\n2. ⚠️  Low Importance Feature Type: {bottom_type[0]}")
            print(f"   - Average importance: {bottom_type[1]:.4f}")
            print(f"   - Recommendation: Consider removing or reducing {bottom_type[0]} features")
            print(f"     to reduce model complexity and potential overfitting")
    
    # 3. Top individual features
    top_features = analysis_result['top_features'][:10]
    if top_features:
        print(f"\n3. 🔝 Top 10 Most Important Individual Features:")
        for i, feat in enumerate(top_features[:10], 1):
            print(f"   {i}. {feat['name'][:70]}")
            print(f"      - Type: {feat['type']}, Importance: {feat['importance']:.6f}")
            print(f"      - Used by {feat['models_count']} model(s)")
    
    # 4. Consistent features
    consistent = consistent_features['consistent_features']
    if consistent:
        top_consistent = sorted(consistent.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n4. ✅ Most Consistent Features (appear in multiple models' top features):")
        for feat_name, count in top_consistent:
            print(f"   - {feat_name[:70]}: appears in {count} model(s)")
        print(f"   - Recommendation: These features are reliable predictors across models")
    
    # 5. Feature engineering suggestions
    print(f"\n5. 💡 Feature Engineering Suggestions:")
    
    # Check if TF-IDF is important
    tfidf_importance = next((t[1] for t in sorted_types if t[0] == 'TF-IDF'), 0)
    if tfidf_importance > 0.2:
        print(f"   - TF-IDF features are highly important ({tfidf_importance:.2%})")
        print(f"     → Consider experimenting with different n-gram ranges or IDF weighting")
    
    # Check if Word2Vec is important
    w2v_importance = next((t[1] for t in sorted_types if t[0] == 'Word2Vec'), 0)
    if w2v_importance > 0.2:
        print(f"   - Word2Vec features are highly important ({w2v_importance:.2%})")
        print(f"     → Consider using pre-trained embeddings or fine-tuning word2vec")
    
    # Check if One-Hot is important
    onehot_importance = next((t[1] for t in sorted_types if t[0] == 'One-Hot'), 0)
    if onehot_importance > 0.15:
        print(f"   - One-Hot encoded categorical features are important ({onehot_importance:.2%})")
        print(f"     → Ensure all relevant categorical variables are properly encoded")
    
    # 6. Model-specific recommendations
    print(f"\n6. 🤖 Model-Specific Observations:")
    for result in all_results:
        model_name = result['name']
        top_type = max(result['type_importance'].items(), key=lambda x: x[1])
        print(f"   - {model_name}: Most important type is {top_type[0]} ({top_type[1]:.4f})")
    
    # 7. Feature reduction suggestion
    total_features = feature_info['total_features']
    if total_features > 1000:
        print(f"\n7. 📉 Feature Reduction Suggestion:")
        print(f"   - Current total features: {total_features:,}")
        print(f"   - Consider feature selection to reduce to top 500-1000 features")
        print(f"   - This may improve model interpretability and reduce overfitting risk")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_feature_importance(input_file: str):
    """Main function to analyze feature importance across all models."""
    if not SKLEARN_AVAILABLE:
        print("\n❌ Error: sklearn not available. Cannot run analysis.")
        return
    
    # Load data
    X, y, feature_cols, feature_info = load_features(input_file)
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
    
    # Initialize all models and extract importance
    print("\n" + "=" * 80)
    print("TRAINING MODELS AND EXTRACTING FEATURE IMPORTANCE")
    print("=" * 80)
    
    if not XGBOOST_AVAILABLE:
        print("\n⚠️  Note: XGBoost is not available. Install with: pip install xgboost")
        print("   Continuing with other models...\n")
    
    all_results = []
    
    # 1. Ridge Regression
    if SKLEARN_AVAILABLE:
        ridge_model = Ridge(alpha=1.0, max_iter=1000)
        result = train_and_extract_importance(
            "Ridge Regression", ridge_model,
            X_train, X_test, y_train, y_test,
            feature_cols, feature_info,
            needs_scaling=True, scaler=scaler
        )
        if result:
            all_results.append(result)
    
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
        result = train_and_extract_importance(
            "MLP Neural Network", mlp_model,
            X_train, X_test, y_train, y_test,
            feature_cols, feature_info,
            needs_scaling=True, scaler=scaler
        )
        if result:
            all_results.append(result)
    
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
        result = train_and_extract_importance(
            "XGBoost", xgb_model,
            X_train, X_test, y_train, y_test,
            feature_cols, feature_info,
            needs_scaling=False
        )
        if result:
            all_results.append(result)
    
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
        result = train_and_extract_importance(
            "Random Forest", rf_model,
            X_train, X_test, y_train, y_test,
            feature_cols, feature_info,
            needs_scaling=False
        )
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No models were successfully trained.")
        return
    
    # Print model results summary
    print("\n" + "=" * 80)
    print("MODEL TRAINING RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Test R²':>12} {'Test RMSE':>15}")
    print("=" * 55)
    for result in all_results:
        print(f"{result['name']:<25} {result['test_r2']:>12.4f} ${result['test_rmse']:>14,.0f}")
    print(f"\n✅ Successfully trained {len(all_results)} model(s)")
    
    # Perform analyses
    analysis_result = analyze_top_features(all_results, feature_cols, feature_info, top_n=30)
    type_analysis = analyze_feature_type_importance(all_results)
    consistent_features = find_consistent_features(all_results, feature_cols, top_n_per_model=20)
    
    # Generate recommendations
    generate_recommendations(all_results, analysis_result, type_analysis, 
                            consistent_features, feature_info)
    
    # Visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n📊 Generating visualizations...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..')
        
        plot_top_features_by_model(all_results, feature_info, output_dir, top_n=20)
        plot_feature_type_importance(all_results, output_dir)
        plot_aggregated_top_features(analysis_result, feature_info, output_dir, top_n=30)
        plot_all_models_top100(all_results, feature_info, output_dir, top_n=100)
        
        print("   ✅ All visualizations saved to project root directory")
    else:
        print("\n⚠️  matplotlib not available. Skipping visualizations.")
    
    print("\n" + "=" * 80)
    print("✅ Feature importance analysis completed!")
    print("=" * 80)


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
        analyze_feature_importance(input_file)
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

