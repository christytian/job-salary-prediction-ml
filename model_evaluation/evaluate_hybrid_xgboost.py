"""
Complete Feature Evaluation with XGBoost (Hybrid NLP + Optimized Categorical Encoding)
================================================================================

This script evaluates the quality and effectiveness of the complete feature set using XGBoost:
- NLP features: TF-IDF + Word2Vec from text fields
- Optimized categorical features: One-Hot, Frequency Encoding, Binary Encoding

Evaluation metrics:
1. Feature quality: feature statistics for all feature types
2. Predictive power: XGBoost model performance with all features
3. Feature importance: Top contributing features from XGBoost
"""

import csv
import numpy as np
import sys
import os
from typing import Dict, List

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Please install: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Model evaluation will be skipped.")

# Text fields that were processed
TEXT_FIELDS = [
    'description',
    'title',
    'all_skills',
    'description_company',
    'all_industries',
    'benefits_list'
]


# =============================================================================
# FEATURE QUALITY ANALYSIS
# =============================================================================

def analyze_feature_quality(data: List[Dict], fieldnames: List[str]) -> Dict:
    """
    Analyze the quality of extracted hybrid features.
    
    Returns:
        Dictionary with quality metrics
    """
    print("\n" + "=" * 80)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 80)
    
    quality_metrics = {}
    
    for field_name in TEXT_FIELDS:
        # Get TF-IDF and Word2Vec feature columns for this field
        tfidf_cols = [col for col in fieldnames 
                     if col.startswith(f'{field_name}_tfidf_')]
        w2v_cols = [col for col in fieldnames 
                    if col.startswith(f'{field_name}_w2v_')]
        
        if not tfidf_cols and not w2v_cols:
            print(f"\n⚠️  Warning: No features found for field '{field_name}'")
            continue
        
        # Extract vectors
        tfidf_features = []
        w2v_features = []
        
        for row in data:
            # TF-IDF features
            tfidf_vec = []
            for col in sorted(tfidf_cols):
                try:
                    val = float(row.get(col, 0))
                    tfidf_vec.append(val)
                except (ValueError, TypeError):
                    tfidf_vec.append(0.0)
            
            # Word2Vec features
            w2v_vec = []
            for col in sorted(w2v_cols):
                try:
                    val = float(row.get(col, 0))
                    w2v_vec.append(val)
                except (ValueError, TypeError):
                    w2v_vec.append(0.0)
            
            if tfidf_vec:
                tfidf_features.append(tfidf_vec)
            if w2v_vec:
                w2v_features.append(w2v_vec)
        
        quality_metrics[field_name] = {
            'tfidf_count': len(tfidf_cols),
            'w2v_count': len(w2v_cols),
            'total_count': len(tfidf_cols) + len(w2v_cols)
        }
        
        if tfidf_features:
            tfidf_array = np.array(tfidf_features)
            quality_metrics[field_name].update({
                'tfidf_mean': np.mean(tfidf_array),
                'tfidf_std': np.std(tfidf_array),
                'tfidf_non_zero_pct': (np.count_nonzero(tfidf_array) / tfidf_array.size) * 100
            })
        
        if w2v_features:
            w2v_array = np.array(w2v_features)
            quality_metrics[field_name].update({
                'w2v_mean': np.mean(w2v_array),
                'w2v_std': np.std(w2v_array),
                'w2v_non_zero_pct': (np.count_nonzero(w2v_array) / w2v_array.size) * 100
            })
        
        # Print results
        print(f"\n📊 Field: {field_name}")
        print(f"   TF-IDF features: {quality_metrics[field_name]['tfidf_count']}")
        print(f"   Word2Vec features: {quality_metrics[field_name]['w2v_count']}")
        print(f"   Total features: {quality_metrics[field_name]['total_count']}")
        if 'tfidf_mean' in quality_metrics[field_name]:
            print(f"   TF-IDF mean: {quality_metrics[field_name]['tfidf_mean']:.6f} ± {quality_metrics[field_name]['tfidf_std']:.6f}")
            print(f"   TF-IDF non-zero: {quality_metrics[field_name]['tfidf_non_zero_pct']:.1f}%")
        if 'w2v_mean' in quality_metrics[field_name]:
            print(f"   Word2Vec mean: {quality_metrics[field_name]['w2v_mean']:.6f} ± {quality_metrics[field_name]['w2v_std']:.6f}")
            print(f"   Word2Vec non-zero: {quality_metrics[field_name]['w2v_non_zero_pct']:.1f}%")
    
    return quality_metrics


# =============================================================================
# PREDICTIVE POWER EVALUATION
# =============================================================================

def evaluate_predictive_power(data: List[Dict], fieldnames: List[str]) -> Dict:
    """
    Evaluate predictive power of hybrid features using XGBoost.
    
    Returns:
        Dictionary with model performance metrics
    """
    if not XGBOOST_AVAILABLE:
        print("\n⚠️  xgboost not available. Skipping predictive power evaluation.")
        print("   Please install: pip install xgboost")
        return {}
    
    if not SKLEARN_AVAILABLE:
        print("\n⚠️  sklearn not available. Skipping predictive power evaluation.")
        return {}
    
    print("\n" + "=" * 80)
    print("PREDICTIVE POWER EVALUATION (XGBoost)")
    print("=" * 80)
    
    # Extract target variable
    target_col = 'salary_normalized'
    if target_col not in fieldnames:
        print(f"\n⚠️  Warning: Target column '{target_col}' not found.")
        return {}
    
    # Prepare data
    X_list = []
    y_list = []
    
    # Get all feature columns (exclude target and non-feature columns)
    # Include: NLP features (TF-IDF, Word2Vec), One-Hot, Frequency, Binary, and numeric features
    excluded_cols = ['salary_normalized', 'normalized_salary', 'benefits_list_missing']
    
    feature_cols = []
    for col in fieldnames:
        if col in excluded_cols:
            continue
        # Include all numeric features (NLP, encoded categorical, and other numeric)
        feature_cols.append(col)
    
    # Count feature types
    tfidf_cols = [c for c in feature_cols if '_tfidf_' in c]
    w2v_cols = [c for c in feature_cols if '_w2v_' in c]
    onehot_cols = [c for c in feature_cols if any(c.startswith(f'{feat}_') for feat in ['formatted_work_type', 'formatted_experience_level', 'company_size'])]
    frequency_cols = [c for c in feature_cols if '_frequency' in c]
    binary_cols = [c for c in feature_cols if '_binary' in c]
    other_numeric_cols = [c for c in feature_cols if c not in tfidf_cols + w2v_cols + onehot_cols + frequency_cols + binary_cols]
    
    print("\n📊 Dataset preparation:")
    print(f"   Total feature columns: {len(feature_cols)}")
    print("   Breakdown:")
    print(f"      - TF-IDF: {len(tfidf_cols)}")
    print(f"      - Word2Vec: {len(w2v_cols)}")
    print(f"      - One-Hot: {len(onehot_cols)}")
    print(f"      - Frequency: {len(frequency_cols)}")
    print(f"      - Binary: {len(binary_cols)}")
    print(f"      - Other numeric: {len(other_numeric_cols)}")
    
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
        print(f"\n⚠️  Warning: Only {len(X_list)} valid samples. Need at least 100 for evaluation.")
        return {}
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"   Valid samples: {len(X):,}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Train XGBoost model
    print("\n🤖 Training XGBoost model...")
    print("   Using default parameters (can be tuned for better performance)")
    
    # XGBoost parameters - can be tuned for better performance
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0  # Suppress XGBoost output
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    print("\n📈 Model Performance:")
    print(f"   Train RMSE: ${train_rmse:,.0f}")
    print(f"   Test RMSE:  ${test_rmse:,.0f}")
    print(f"   Train MAE:  ${train_mae:,.0f}")
    print(f"   Test MAE:   ${test_mae:,.0f}")
    print(f"   Train R²:   {train_r2:.4f}")
    print(f"   Test R²:    {test_r2:.4f}")
    
    # Feature importance (XGBoost provides feature_importances_)
    feature_importance = model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-20:][::-1]  # Top 20 features
    
    print("\n🔝 Top 20 Most Important Features (by XGBoost importance):")
    for idx in top_features_idx:
        print(f"   {feature_cols[idx]}: {feature_importance[idx]:.6f}")
    
    return results


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_hybrid_features(input_file: str):
    """
    Comprehensive evaluation of complete feature set (NLP + Optimized Categorical) using XGBoost.
    
    Args:
        input_file: Path to CSV file with all features (NLP + optimized categorical encoding)
    """
    print("=" * 80)
    print("COMPLETE FEATURE EVALUATION WITH XGBOOST")
    print("(NLP Features + Optimized Categorical Encoding)")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    
    # Read data
    print("\n📖 Reading data file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"   Read {len(data):,} records")
    print(f"   Total columns: {len(fieldnames) if fieldnames else 0}")
    
    # Count feature columns by type
    tfidf_cols = [col for col in fieldnames if '_tfidf_' in col] if fieldnames else []
    w2v_cols = [col for col in fieldnames if '_w2v_' in col] if fieldnames else []
    onehot_cols = [col for col in fieldnames if any(col.startswith(f'{feat}_') for feat in ['formatted_work_type', 'formatted_experience_level', 'company_size'])] if fieldnames else []
    frequency_cols = [col for col in fieldnames if '_frequency' in col] if fieldnames else []
    binary_cols = [col for col in fieldnames if '_binary' in col] if fieldnames else []
    
    print("   Feature breakdown:")
    print(f"      - TF-IDF: {len(tfidf_cols)}")
    print(f"      - Word2Vec: {len(w2v_cols)}")
    print(f"      - One-Hot: {len(onehot_cols)}")
    print(f"      - Frequency: {len(frequency_cols)}")
    print(f"      - Binary: {len(binary_cols)}")
    print(f"      - Total NLP: {len(tfidf_cols) + len(w2v_cols)}")
    print(f"      - Total categorical (encoded): {len(onehot_cols) + len(frequency_cols) + len(binary_cols)}")
    
    # Run evaluations
    quality_metrics = analyze_feature_quality(data, fieldnames)
    predictive_results = evaluate_predictive_power(data, fieldnames)
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\n✅ Feature Quality:")
    for field_name, metrics in quality_metrics.items():
        print(f"   {field_name}: {metrics['tfidf_count']} TF-IDF + {metrics['w2v_count']} Word2Vec = {metrics['total_count']} total")
    
    if predictive_results:
        print("\n✅ Predictive Power (XGBoost):")
        print(f"   Test R²: {predictive_results['test_r2']:.4f}")
        print(f"   Test RMSE: ${predictive_results['test_rmse']:,.0f}")
        print(f"   Test MAE: ${predictive_results['test_mae']:,.0f}")
        
        print("\n📊 Model Performance Summary:")
        print("   All features (Hybrid NLP + Optimized Categorical) with XGBoost:")
        print(f"   Test R²: {predictive_results['test_r2']:.4f}")
        print(f"   Test RMSE: ${predictive_results['test_rmse']:,.0f}")
        print(f"   Test MAE: ${predictive_results['test_mae']:,.0f}")
        print(f"   Features used: {predictive_results['n_features']:,}")
        print(f"   Samples: {predictive_results['n_samples']:,}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


def main():
    """Main entry point."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    # Default input file (optimized with all features)
    input_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_onehot_features.csv')
    
    # Allow command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    try:
        evaluate_hybrid_features(input_file)
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {input_file}")
        print("   Please run onehot_encode.py first to generate optimized features.")
        sys.exit(1)
    except (OSError, KeyError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

