"""
Train Production MLP Model for Salary Prediction
==================================================

This script trains the final production MLP (Multi-Layer Perceptron) model 
for salary prediction based on the best-performing model configuration.

Based on comprehensive model comparison, MLP Neural Network achieved:
- Test R²: 0.6739 (explains 67.39% of salary variance)
- Test RMSE: $31,557
- Test MAE: $21,234

Features:
- NLP features: TF-IDF + Word2Vec from text fields
- Optimized categorical features: One-Hot, Frequency Encoding, Binary Encoding

Outputs:
- Trained MLP model (saved as joblib file)
- StandardScaler (saved as joblib file)
- Feature column names (saved as JSON file)
- Model metadata and performance report
"""

import csv
import numpy as np
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"Error: Required library not available: {e}")
    print("Please install: pip install scikit-learn joblib")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# MLP Model Configuration (best performing from model comparison)
MLP_CONFIG = {
    'hidden_layer_sizes': (128, 64),    # Two hidden layers: 128 and 64 neurons
    'activation': 'relu',                # ReLU activation function
    'solver': 'adam',                    # Adam optimizer
    'alpha': 0.001,                      # L2 regularization parameter
    'learning_rate': 'adaptive',         # Adaptive learning rate
    'learning_rate_init': 0.001,         # Initial learning rate
    'max_iter': 500,                     # Maximum iterations
    'early_stopping': True,              # Stop if validation score doesn't improve
    'validation_fraction': 0.1,          # Fraction of training data for validation
    'n_iter_no_change': 20,              # Stop if no improvement for 20 iterations
    'random_state': 42                   # For reproducibility
}

# Train/Test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Excluded columns (not used as features)
EXCLUDED_COLS = ['salary_normalized', 'normalized_salary', 'benefits_list_missing']


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data(input_file: str) -> Tuple[List[Dict], List[str]]:
    """
    Load CSV data file.
    
    Args:
        input_file: Path to input CSV file
        
    Returns:
        Tuple of (data records, fieldnames)
    """
    print(f"\n📖 Loading data from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = list(reader.fieldnames)
    
    print(f"   ✅ Loaded: {len(data):,} records, {len(fieldnames)} columns")
    
    return data, fieldnames


def prepare_features_and_target(data: List[Dict], fieldnames: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix (X) and target vector (y) from data.
    
    Args:
        data: List of data records
        fieldnames: List of column names
        
    Returns:
        Tuple of (X, y, feature_cols)
    """
    target_col = 'salary_normalized'
    
    if target_col not in fieldnames:
        raise ValueError(f"Target column '{target_col}' not found in data!")
    
    # Get all feature columns (exclude target and excluded columns)
    feature_cols = []
    for col in fieldnames:
        if col in EXCLUDED_COLS:
            continue
        feature_cols.append(col)
    
    # Count feature types for reporting
    tfidf_cols = [c for c in feature_cols if '_tfidf_' in c]
    w2v_cols = [c for c in feature_cols if '_w2v_' in c]
    onehot_cols = [c for c in feature_cols if any(c.startswith(f'{feat}_') for feat in ['formatted_work_type', 'formatted_experience_level', 'company_size'])]
    frequency_cols = [c for c in feature_cols if '_frequency' in c]
    binary_cols = [c for c in feature_cols if '_binary' in c]
    other_numeric_cols = [c for c in feature_cols if c not in tfidf_cols + w2v_cols + onehot_cols + frequency_cols + binary_cols]
    
    print(f"\n📊 Feature Preparation:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Breakdown:")
    print(f"      - TF-IDF: {len(tfidf_cols)}")
    print(f"      - Word2Vec: {len(w2v_cols)}")
    print(f"      - One-Hot: {len(onehot_cols)}")
    print(f"      - Frequency: {len(frequency_cols)}")
    print(f"      - Binary: {len(binary_cols)}")
    print(f"      - Other numeric: {len(other_numeric_cols)}")
    
    # Extract features and target
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
        raise ValueError(f"Only {len(X_list)} valid samples. Need at least 100 for training.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"   ✅ Valid samples: {len(X):,}")
    
    return X, y, feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_mlp_model(X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray,
                    scaler: StandardScaler) -> Tuple[MLPRegressor, Dict]:
    """
    Train MLP model and return trained model with performance metrics.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        scaler: Fitted StandardScaler
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🤖 Training MLP Neural Network model...")
    print(f"   Architecture: {MLP_CONFIG['hidden_layer_sizes']} hidden layers")
    print(f"   Activation: {MLP_CONFIG['activation']}")
    print(f"   Solver: {MLP_CONFIG['solver']}")
    print(f"   Regularization (alpha): {MLP_CONFIG['alpha']}")
    
    # Create and train model
    model = MLPRegressor(**MLP_CONFIG)
    
    model.fit(X_train_scaled, y_train)
    
    # Check convergence
    if model.n_iter_ == MLP_CONFIG['max_iter']:
        print(f"   ⚠️  Warning: Training reached max_iter ({MLP_CONFIG['max_iter']})")
    else:
        print(f"   ✅ Training converged after {model.n_iter_} iterations")
    
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
    
    metrics = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'n_iter': int(model.n_iter_),
        'loss': str(model.loss_)
    }
    
    print(f"\n📈 Model Performance:")
    print(f"   Train R²:   {train_r2:.4f}")
    print(f"   Test R²:    {test_r2:.4f}")
    print(f"   Train RMSE: ${train_rmse:,.0f}")
    print(f"   Test RMSE:  ${test_rmse:,.0f}")
    print(f"   Train MAE:  ${train_mae:,.0f}")
    print(f"   Test MAE:   ${test_mae:,.0f}")
    
    # Calculate overfitting gap
    overfitting_gap = train_r2 - test_r2
    print(f"\n   Overfitting gap: {overfitting_gap:.4f}")
    if overfitting_gap < 0.05:
        print("   ✅ No significant overfitting detected")
    elif overfitting_gap < 0.15:
        print("   ⚠️  Moderate overfitting detected")
    else:
        print("   🔴 Significant overfitting detected - consider regularization")
    
    return model, metrics


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_model_artifacts(model: MLPRegressor, scaler: StandardScaler, 
                         feature_cols: List[str], metrics: Dict,
                         output_dir: str) -> Dict[str, str]:
    """
    Save model artifacts to disk.
    
    Args:
        model: Trained MLP model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        metrics: Model performance metrics
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary mapping artifact names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    artifact_paths = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(output_dir, 'mlp_salary_model.joblib')
    joblib.dump(model, model_path)
    artifact_paths['model'] = model_path
    print(f"\n💾 Saved model: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'mlp_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    artifact_paths['scaler'] = scaler_path
    print(f"💾 Saved scaler: {scaler_path}")
    
    # Save feature columns
    feature_cols_path = os.path.join(output_dir, 'mlp_feature_columns.json')
    with open(feature_cols_path, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=2)
    artifact_paths['feature_columns'] = feature_cols_path
    print(f"💾 Saved feature columns: {feature_cols_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'MLPRegressor',
        'model_config': MLP_CONFIG,
        'training_date': datetime.now().isoformat(),
        'performance_metrics': metrics,
        'n_features': len(feature_cols),
        'feature_columns_file': 'mlp_feature_columns.json',
        'model_file': 'mlp_salary_model.joblib',
        'scaler_file': 'mlp_scaler.joblib'
    }
    
    metadata_path = os.path.join(output_dir, 'mlp_model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    artifact_paths['metadata'] = metadata_path
    print(f"💾 Saved model metadata: {metadata_path}")
    
    return artifact_paths


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_production_model(input_file: str, output_dir: str = None) -> Dict:
    """
    Train production MLP model for salary prediction.
    
    Args:
        input_file: Path to input CSV file with features
        output_dir: Directory to save model artifacts (default: model_evaluation/models/)
        
    Returns:
        Dictionary with training results and artifact paths
    """
    print("=" * 80)
    print("TRAIN PRODUCTION MLP MODEL FOR SALARY PREDICTION")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'models')
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    data, fieldnames = load_data(input_file)
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(data, fieldnames)
    
    # Train/Test split
    print(f"\n📊 Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Scale features
    print("\n📏 Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("   ✅ Scaler fitted")
    
    # Train model
    model, metrics = train_mlp_model(X_train, X_test, y_train, y_test, scaler)
    
    # Save model artifacts
    print("\n" + "=" * 80)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 80)
    
    artifact_paths = save_model_artifacts(
        model, scaler, feature_cols, metrics, output_dir
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Model Training Completed Successfully!")
    print(f"\n📊 Performance Metrics:")
    print(f"   Test R²:    {metrics['test_r2']:.4f} (explains {metrics['test_r2']*100:.2f}% of variance)")
    print(f"   Test RMSE:  ${metrics['test_rmse']:,.0f}")
    print(f"   Test MAE:   ${metrics['test_mae']:,.0f}")
    
    print(f"\n📁 Saved Artifacts:")
    for artifact_name, artifact_path in artifact_paths.items():
        rel_path = os.path.relpath(artifact_path, output_dir)
        print(f"   • {artifact_name}: {rel_path}")
    
    print(f"\n💡 Usage Example:")
    print(f"   import joblib")
    print(f"   import json")
    print(f"   ")
    print(f"   # Load model")
    print(f"   model = joblib.load('{os.path.relpath(artifact_paths['model'], '.')}')")
    print(f"   scaler = joblib.load('{os.path.relpath(artifact_paths['scaler'], '.')}')")
    print(f"   with open('{os.path.relpath(artifact_paths['feature_columns'], '.')}', 'r') as f:")
    print(f"       feature_cols = json.load(f)")
    print(f"   ")
    print(f"   # Make prediction")
    print(f"   X_scaled = scaler.transform([your_features])")
    print(f"   salary_prediction = model.predict(X_scaled)[0]")
    
    print("\n" + "=" * 80)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'artifact_paths': artifact_paths
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Default input file
    input_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_onehot_features.csv')
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # Optional output directory
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    try:
        train_production_model(input_file, output_dir)
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {input_file}")
        print("   Please run feature_encoding_all.py first to generate optimized features.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

