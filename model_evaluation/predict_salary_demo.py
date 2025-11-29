"""
Salary Prediction Demo Script
==============================

This script demonstrates how to use the trained MLP model to predict salaries.

Usage Examples:
1. Predict salary from a row in the feature CSV file
2. Predict salary from a dictionary of features
3. Batch prediction for multiple records

Note: For production use, you would need to prepare features from raw job data
using the full feature engineering pipeline (NLP + encoding).
"""

import csv
import json
import numpy as np
import sys
import os
from typing import Dict, List, Optional

try:
    import joblib
except ImportError:
    print("Error: joblib not installed. Please install: pip install joblib")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model artifact paths (relative to script directory)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mlp_salary_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'mlp_scaler.joblib')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'mlp_feature_columns.json')
METADATA_PATH = os.path.join(MODEL_DIR, 'mlp_model_metadata.json')


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model() -> tuple:
    """
    Load the trained model, scaler, and feature columns.
    
    Returns:
        Tuple of (model, scaler, feature_cols, metadata)
    """
    print("📦 Loading model artifacts...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURE_COLS_PATH, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"   ✅ Model loaded: {os.path.basename(MODEL_PATH)}")
    print(f"   ✅ Scaler loaded: {os.path.basename(SCALER_PATH)}")
    print(f"   ✅ Feature columns: {len(feature_cols)} features")
    print(f"   ✅ Model performance: Test R² = {metadata['performance_metrics']['test_r2']:.4f}")
    
    return model, scaler, feature_cols, metadata


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_from_feature_dict(feature_dict: Dict[str, float], 
                              model, scaler, feature_cols: List[str]) -> float:
    """
    Predict salary from a dictionary of features.
    
    Args:
        feature_dict: Dictionary mapping feature names to values
        model: Trained MLP model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names (in correct order)
        
    Returns:
        Predicted salary
    """
    # Create feature vector in correct order
    features = []
    missing_features = []
    
    for col in feature_cols:
        if col in feature_dict:
            features.append(float(feature_dict[col]))
        else:
            features.append(0.0)  # Default to 0 for missing features
            missing_features.append(col)
    
    # Suppress missing features warning for cleaner output (uncomment to enable)
    # if missing_features:
    #     print(f"   ⚠️  Warning: {len(missing_features)} features missing, using default value 0.0")
    #     if len(missing_features) <= 10:
    #         print(f"      Missing: {', '.join(missing_features[:10])}")
    
    # Convert to numpy array and reshape
    X = np.array(features).reshape(1, -1)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    return float(prediction)


def load_csv_rows(csv_file: str) -> List[Dict]:
    """
    Load all rows from CSV file (optimized - reads once).
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        List of row dictionaries
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def predict_from_row_dict(row: Dict[str, str], row_index: int,
                          model, scaler, feature_cols: List[str]) -> Dict:
    """
    Predict salary from a row dictionary (already loaded from CSV).
    
    Args:
        row: Dictionary representing a CSV row
        row_index: Index of the row (for reporting)
        model: Trained MLP model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with prediction and original data
    """
    # Get actual salary if available
    actual_salary = None
    if 'salary_normalized' in row:
        try:
            actual_salary = float(row['salary_normalized'])
        except (ValueError, TypeError):
            pass
    
    # Extract features and convert to float
    feature_dict_float = {}
    for col in feature_cols:
        val = row.get(col, '0')
        try:
            feature_dict_float[col] = float(val)
        except (ValueError, TypeError):
            feature_dict_float[col] = 0.0
    
    # Predict
    predicted_salary = predict_from_feature_dict(feature_dict_float, model, scaler, feature_cols)
    
    return {
        'row_index': row_index,
        'predicted_salary': predicted_salary,
        'actual_salary': actual_salary,
        'error': abs(predicted_salary - actual_salary) if actual_salary else None,
        'error_pct': abs((predicted_salary - actual_salary) / actual_salary * 100) if actual_salary else None,
        'title': row.get('title', 'N/A')[:60] if 'title' in row else 'N/A',
        'company': row.get('company_name', 'N/A')[:40] if 'company_name' in row else 'N/A'
    }


def predict_from_csv_row(csv_file: str, row_index: int = 0,
                         model=None, scaler=None, feature_cols=None,
                         rows: List[Dict] = None) -> Dict:
    """
    Predict salary from a row in the feature CSV file.
    
    Args:
        csv_file: Path to CSV file with features
        row_index: Index of row to predict (default: 0)
        model: Trained model (will load if None)
        scaler: Scaler (will load if None)
        feature_cols: Feature columns (will load if None)
        rows: Pre-loaded rows list (optional, will read file if not provided)
        
    Returns:
        Dictionary with prediction and original data
    """
    # Load model if not provided
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols, metadata = load_model()
    
    # Read CSV file if rows not provided
    if rows is None:
        print(f"\n📖 Reading CSV file: {csv_file}")
        rows = load_csv_rows(csv_file)
    
    if row_index >= len(rows):
        raise IndexError(f"Row index {row_index} out of range. File has {len(rows)} rows.")
    
    row = rows[row_index]
    
    # Predict from row
    return predict_from_row_dict(row, row_index, model, scaler, feature_cols)


def batch_predict_rows(rows: List[Dict], row_indices: List[int],
                       model, scaler, feature_cols: List[str]) -> tuple:
    """
    Batch predict salaries for multiple rows (vectorized, efficient).
    
    Args:
        rows: List of row dictionaries
        row_indices: List of row indices to predict
        model: Trained MLP model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (predictions array, valid_indices list)
    """
    # Prepare feature matrix
    X_list = []
    valid_indices = []
    
    for idx in row_indices:
        if idx >= len(rows):
            continue
        
        row = rows[idx]
        features = []
        
        for col in feature_cols:
            val = row.get(col, '0')
            try:
                features.append(float(val))
            except (ValueError, TypeError):
                features.append(0.0)
        
        X_list.append(features)
        valid_indices.append(idx)
    
    if len(X_list) == 0:
        return np.array([]), []
    
    # Convert to numpy array
    X = np.array(X_list)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Batch predict (much faster than individual predictions)
    predictions = model.predict(X_scaled)
    
    return predictions, valid_indices


def batch_predict(csv_file: str, n_samples: int = 10,
                  model=None, scaler=None, feature_cols=None,
                  rows: List[Dict] = None) -> List[Dict]:
    """
    Predict salaries for multiple rows in CSV file (optimized - reads CSV only once, uses batch prediction).
    
    Args:
        csv_file: Path to CSV file with features
        n_samples: Number of samples to predict (default: 10)
        model: Trained model (will load if None)
        scaler: Scaler (will load if None)
        feature_cols: Feature columns (will load if None)
        rows: Pre-loaded rows list (optional, will read file if not provided)
        
    Returns:
        List of prediction dictionaries
    """
    # Load model if not provided
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols, metadata = load_model()
    
    # Read CSV file once if rows not provided
    if rows is None:
        print(f"\n📖 Reading CSV file: {csv_file}")
        rows = load_csv_rows(csv_file)
        print(f"   ✅ Loaded {len(rows):,} rows")
    
    # Limit n_samples to available rows
    n_samples = min(n_samples, len(rows))
    
    print(f"\n📊 Batch prediction for {n_samples} samples (vectorized)...")
    
    # Get row indices
    row_indices = list(range(n_samples))
    
    # Batch predict (vectorized - much faster)
    predictions, valid_indices = batch_predict_rows(rows, row_indices, model, scaler, feature_cols)
    
    # Build results
    results = []
    for pred_idx, row_idx in enumerate(valid_indices):
        row = rows[row_idx]
        
        # Get actual salary if available
        actual_salary = None
        if 'salary_normalized' in row:
            try:
                actual_salary = float(row['salary_normalized'])
            except (ValueError, TypeError):
                pass
        
        predicted_salary = float(predictions[pred_idx])
        
        result = {
            'row_index': row_idx,
            'predicted_salary': predicted_salary,
            'actual_salary': actual_salary,
            'error': abs(predicted_salary - actual_salary) if actual_salary else None,
            'error_pct': abs((predicted_salary - actual_salary) / actual_salary * 100) if actual_salary else None,
            'title': row.get('title', 'N/A')[:60] if 'title' in row else 'N/A',
            'company': row.get('company_name', 'N/A')[:40] if 'company_name' in row else 'N/A'
        }
        results.append(result)
    
    print(f"   ✅ Predicted {len(results)} salaries")
    
    return results


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demo_single_prediction():
    """Demonstrate single salary prediction from CSV file."""
    print("=" * 80)
    print("DEMO 1: SINGLE SALARY PREDICTION")
    print("=" * 80)
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    csv_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_onehot_features.csv')
    
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: Feature CSV file not found: {csv_file}")
        print("   Please run the feature extraction pipeline first.")
        return
    
    # Load model
    model, scaler, feature_cols, metadata = load_model()
    
    # Load CSV rows (only once)
    print(f"\n📖 Reading CSV file: {csv_file}")
    rows = load_csv_rows(csv_file)
    print(f"   ✅ Loaded {len(rows):,} rows")
    
    # Predict for first row
    print(f"\n🔮 Predicting salary for row 0...")
    result = predict_from_row_dict(rows[0], 0, model, scaler, feature_cols)
    
    print(f"\n📊 Prediction Result:")
    print(f"   Job Title:  {result['title']}")
    print(f"   Company:    {result['company']}")
    print(f"   Predicted Salary: ${result['predicted_salary']:,.0f}")
    
    if result['actual_salary']:
        print(f"   Actual Salary:    ${result['actual_salary']:,.0f}")
        print(f"   Error:            ${result['error']:,.0f} ({result['error_pct']:.1f}%)")
    
    return result


def demo_batch_prediction():
    """Demonstrate batch salary prediction (optimized - reads CSV only once)."""
    print("\n" + "=" * 80)
    print("DEMO 2: BATCH SALARY PREDICTION")
    print("=" * 80)
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    csv_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_onehot_features.csv')
    
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: Feature CSV file not found: {csv_file}")
        return
    
    # Load model
    model, scaler, feature_cols, metadata = load_model()
    
    # Load CSV rows once (optimized)
    print(f"\n📖 Reading CSV file: {csv_file}")
    rows = load_csv_rows(csv_file)
    print(f"   ✅ Loaded {len(rows):,} rows")
    
    # Batch predict (pass pre-loaded rows)
    results = batch_predict(csv_file, n_samples=50, 
                            model=model, scaler=scaler, feature_cols=feature_cols,
                            rows=rows)
    
    print(f"\n📊 Batch Prediction Results ({len(results)} samples):")
    print(f"\n{'Index':<8} {'Predicted':<15} {'Actual':<15} {'Error':<15} {'Error %':<10}")
    print("-" * 75)
    
    for r in results:
        actual_str = f"${r['actual_salary']:,.0f}" if r['actual_salary'] else "N/A"
        error_str = f"${r['error']:,.0f}" if r['error'] else "N/A"
        error_pct_str = f"{r['error_pct']:.1f}%" if r['error_pct'] else "N/A"
        
        print(f"{r['row_index']:<8} "
              f"${r['predicted_salary']:>12,.0f}  "
              f"{actual_str:>13}  "
              f"{error_str:>13}  "
              f"{error_pct_str:>9}")
    
    # Calculate average error
    errors = [r['error'] for r in results if r['error'] is not None]
    if errors:
        avg_error = np.mean(errors)
        avg_error_pct = np.mean([r['error_pct'] for r in results if r['error_pct'] is not None])
        print("-" * 75)
        print(f"Average Error: ${avg_error:,.0f} ({avg_error_pct:.1f}%)")
    
    return results


def demo_usage_example():
    """Show usage example with code."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLE CODE")
    print("=" * 80)
    
    print("""
# Example 1: Load model and predict from feature dictionary

import joblib
import json
import numpy as np

# Load model artifacts
model = joblib.load('models/mlp_salary_model.joblib')
scaler = joblib.load('models/mlp_scaler.joblib')
with open('models/mlp_feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

# Prepare features (in correct order)
features = [0.0] * len(feature_cols)  # Initialize with zeros
# Set your feature values here based on feature_cols order
# features[0] = your_value_for_feature_cols[0]
# features[1] = your_value_for_feature_cols[1]
# ... etc

# Convert to numpy and reshape
X = np.array(features).reshape(1, -1)

# Scale features
X_scaled = scaler.transform(X)

# Predict salary
predicted_salary = model.predict(X_scaled)[0]
print(f"Predicted Salary: ${predicted_salary:,.0f}")

# Example 2: Predict from CSV row
from predict_salary_demo import predict_from_csv_row

result = predict_from_csv_row('salary_data_with_nlp_hybrid_onehot_features.csv', row_index=0)
print(f"Predicted: ${result['predicted_salary']:,.0f}")
print(f"Actual: ${result['actual_salary']:,.0f}")
    """)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run prediction demos."""
    print("=" * 80)
    print("SALARY PREDICTION DEMO")
    print("=" * 80)
    print("\nThis script demonstrates how to use the trained MLP model")
    print("to predict job salaries from feature data.")
    
    try:
        # Demo 1: Single prediction
        demo_single_prediction()
        
        # Demo 2: Batch prediction
        demo_batch_prediction()
        
        # Show usage example
        demo_usage_example()
        
        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure:")
        print("  1. Model files are in model_evaluation/models/ directory")
        print("  2. Feature CSV file exists")
        print("  3. Run train_production_mlp.py first to train the model")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

