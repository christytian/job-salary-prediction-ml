"""
Train and Compare NLP Methods with SBERT PCA
=============================================
Combined preprocessing and training script for NLP feature comparison.

This script:
1. Loads feature files for each NLP method (TF-IDF, SBERT PCA, Hybrid)
2. Performs train/test split (80/20)
3. Scales features (StandardScaler for Linear/MLP models)
4. Trains multiple models on each dataset:
   - Linear Regression (baseline)
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
   - XGBoost (with L1/L2 regularization)
   - LightGBM (faster gradient boosting)
   - CatBoost (robust boosting)
   - Random Forest (with regularization parameters)
   - Neural Network (with L2 regularization)
5. Compares performance metrics (R², RMSE, MAE) across all combinations
6. Identifies the best NLP approach and model combination

Input files:
- nlp_features_pca/salary_data_nlp_tfidf.csv
- nlp_features_pca/salary_data_nlp_sbert_pca.csv
- nlp_features_pca/salary_data_nlp_hybrid.csv

Output:
- Comparison report with performance metrics for all combinations
- Saved models (optional)
"""

import csv
import numpy as np
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Will skip XGBoost models.")

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Will skip LightGBM models.")

# Try to import CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Will skip CatBoost models.")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "input_dir": "nlp_features_pca",
    "output_dir": "models_nlp_pca_comparison",
    "random_state": 42,
    "test_size": 0.2,  # 80/20 train/test split

    # Linear Regression (baseline)
    "linear": {
        "fit_intercept": True,
    },

    # Ridge Regression (L2 regularization)
    "ridge": {
        "alpha": 1.0,  # L2 regularization strength
        "fit_intercept": True,
    },

    # Lasso Regression (L1 regularization)
    "lasso": {
        "alpha": 0.1,  # L1 regularization strength (smaller for stability)
        "fit_intercept": True,
        "max_iter": 2000,  # Lasso may need more iterations
    },

    # XGBoost parameters (with regularization)
    "xgb": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
    },

    # LightGBM parameters (with regularization)
    "lgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "num_leaves": 31,
        "min_child_samples": 20,
    },

    # CatBoost parameters (with regularization)
    "catboost": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,  # L2 regularization
        "random_strength": 1.0,  # Additional regularization
        "subsample": 0.8,
    },

    # Random Forest parameters (with regularization)
    "rf": {
        "n_estimators": 100,
        "max_depth": 15,  # Reduced from 20 to reduce overfitting
        "max_features": "sqrt",
        "min_samples_split": 5,  # Increased from 2 (regularization)
        "min_samples_leaf": 3,  # Increased from 1 (regularization)
    },

    # Neural Network parameters (with L2 regularization)
    "nn": {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "adaptive",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "alpha": 0.001,  # L2 regularization strength
    },

    # General
    "n_jobs": -1,  # Use all CPU cores
}

# NLP methods to compare
NLP_METHODS = ["tfidf", "sbert_pca", "hybrid"]

# Models to train
MODELS_TO_TRAIN = [
    "linear",
    "ridge",
    "lasso",
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "neural_network",
]


def load_feature_file(filepath):
    """Load features from CSV file."""
    print(f"   Loading {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Read all data
        data = []
        for row in reader:
            data.append([float(row[col]) for col in fieldnames])

    X = np.array(data)
    feature_names = fieldnames

    print(f"      ✓ Loaded {len(X):,} records × {len(feature_names)} features")

    return X, feature_names


def preprocess_data(X, nlp_method):
    """
    Preprocess data: train/test split and optional scaling.

    Note: Features are already numeric (one-hot encoded + TF-IDF + SBERT PCA),
    so no categorical encoding is needed.
    """
    print(f"\n📊 Preprocessing {nlp_method.upper()} data...")

    # Train/test split
    print(f"   Performing train/test split ({int((1-CONFIG['test_size'])*100)}/{int(CONFIG['test_size']*100)})...")
    X_train, X_test = train_test_split(
        X,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )

    print(f"      ✓ Training set: {len(X_train):,} samples")
    print(f"      ✓ Test set: {len(X_test):,} samples")

    return X_train, X_test


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def create_model(model_type):
    """Create a model instance based on model type."""
    if model_type == "linear":
        return LinearRegression(**CONFIG["linear"])

    elif model_type == "ridge":
        return Ridge(
            alpha=CONFIG["ridge"]["alpha"],
            fit_intercept=CONFIG["ridge"]["fit_intercept"],
            random_state=CONFIG["random_state"],
        )

    elif model_type == "lasso":
        return Lasso(
            alpha=CONFIG["lasso"]["alpha"],
            fit_intercept=CONFIG["lasso"]["fit_intercept"],
            max_iter=CONFIG["lasso"]["max_iter"],
            random_state=CONFIG["random_state"],
        )

    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            return None
        return xgb.XGBRegressor(
            n_estimators=CONFIG["xgb"]["n_estimators"],
            max_depth=CONFIG["xgb"]["max_depth"],
            learning_rate=CONFIG["xgb"]["learning_rate"],
            subsample=CONFIG["xgb"]["subsample"],
            colsample_bytree=CONFIG["xgb"]["colsample_bytree"],
            reg_alpha=CONFIG["xgb"]["reg_alpha"],  # L1 regularization
            reg_lambda=CONFIG["xgb"]["reg_lambda"],  # L2 regularization
            random_state=CONFIG["random_state"],
            n_jobs=CONFIG["n_jobs"],
            verbosity=0,
        )

    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            return None
        return lgb.LGBMRegressor(
            n_estimators=CONFIG["lgbm"]["n_estimators"],
            max_depth=CONFIG["lgbm"]["max_depth"],
            learning_rate=CONFIG["lgbm"]["learning_rate"],
            subsample=CONFIG["lgbm"]["subsample"],
            colsample_bytree=CONFIG["lgbm"]["colsample_bytree"],
            reg_alpha=CONFIG["lgbm"]["reg_alpha"],  # L1 regularization
            reg_lambda=CONFIG["lgbm"]["reg_lambda"],  # L2 regularization
            num_leaves=CONFIG["lgbm"]["num_leaves"],
            min_child_samples=CONFIG["lgbm"]["min_child_samples"],
            random_state=CONFIG["random_state"],
            n_jobs=CONFIG["n_jobs"],
            verbosity=-1,  # Suppress output
        )

    elif model_type == "catboost":
        if not CATBOOST_AVAILABLE:
            return None
        return cb.CatBoostRegressor(
            iterations=CONFIG["catboost"]["iterations"],
            depth=CONFIG["catboost"]["depth"],
            learning_rate=CONFIG["catboost"]["learning_rate"],
            l2_leaf_reg=CONFIG["catboost"]["l2_leaf_reg"],  # L2 regularization
            random_strength=CONFIG["catboost"]["random_strength"],
            subsample=CONFIG["catboost"]["subsample"],
            random_state=CONFIG["random_state"],
            verbose=False,  # Suppress output
        )

    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=CONFIG["rf"]["n_estimators"],
            max_depth=CONFIG["rf"]["max_depth"],
            max_features=CONFIG["rf"]["max_features"],
            min_samples_split=CONFIG["rf"]["min_samples_split"],  # Regularization
            min_samples_leaf=CONFIG["rf"]["min_samples_leaf"],  # Regularization
            random_state=CONFIG["random_state"],
            n_jobs=CONFIG["n_jobs"],
        )

    elif model_type == "neural_network":
        return MLPRegressor(
            hidden_layer_sizes=CONFIG["nn"]["hidden_layer_sizes"],
            activation=CONFIG["nn"]["activation"],
            solver=CONFIG["nn"]["solver"],
            learning_rate=CONFIG["nn"]["learning_rate"],
            max_iter=CONFIG["nn"]["max_iter"],
            early_stopping=CONFIG["nn"]["early_stopping"],
            validation_fraction=CONFIG["nn"]["validation_fraction"],
            n_iter_no_change=CONFIG["nn"]["n_iter_no_change"],
            alpha=CONFIG["nn"]["alpha"],  # L2 regularization
            random_state=CONFIG["random_state"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_name(model_type):
    """Get display name for model type."""
    names = {
        "linear": "Linear Regression",
        "ridge": "Ridge Regression (L2)",
        "lasso": "Lasso Regression (L1)",
        "xgboost": "XGBoost (Regularized)",
        "lightgbm": "LightGBM (Regularized)",
        "catboost": "CatBoost (Regularized)",
        "random_forest": "Random Forest (Regularized)",
        "neural_network": "Neural Network (L2 Regularized)",
    }
    return names.get(model_type, model_type)


def train_model(X_train, y_train, X_test, y_test, nlp_method, model_type):
    """Train a specific model and return evaluation metrics."""
    model = create_model(model_type)

    if model is None:
        return None

    model_name = get_model_name(model_type)

    # Determine if scaling is needed
    needs_scaling = model_type in ["linear", "ridge", "lasso", "neural_network"]

    if needs_scaling:
        print(f"   Applying StandardScaler...")
        X_train_model, X_test_model, scaler = scale_features(X_train, X_test)
    else:
        X_train_model, X_test_model = X_train, X_test
        scaler = None

    # Train
    start_time = time.time()
    try:
        model.fit(X_train_model, y_train)
        train_time = time.time() - start_time
    except Exception as e:
        print(f"   ❌ Error training {model_name}: {e}")
        return None

    # Predict
    y_train_pred = model.predict(X_train_model)
    y_test_pred = model.predict(X_test_model)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Overfitting check
    overfit_diff = train_r2 - test_r2

    # Percentage errors
    mean_salary = np.mean(y_test)
    mae_pct = (test_mae / mean_salary) * 100

    return {
        "nlp_method": nlp_method,
        "model_type": model_type,
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "overfit_diff": overfit_diff,
        "mae_pct": mae_pct,
        "mean_salary": mean_salary,
        "train_time": train_time,
        "n_features": X_train.shape[1],
    }


def print_detailed_results(result):
    """Print detailed results for a single model."""
    print(f"\n   📊 Performance Metrics:")
    print(f"   {'Metric':<20} {'Train':<15} {'Test':<15}")
    print(f"   {'-'*50}")
    print(f"   {'R² Score':<20} {result['train_r2']:<15.4f} {result['test_r2']:<15.4f}")
    print(f"   {'RMSE ($)':<20} ${result['train_rmse']:<14,.0f} ${result['test_rmse']:<14,.0f}")
    print(f"   {'MAE ($)':<20} ${result['train_mae']:<14,.0f} ${result['test_mae']:<14,.0f}")

    print(f"\n   📈 Overfitting Check:")
    print(f"      R² difference (Train - Test): {result['overfit_diff']:.4f}")
    if result['overfit_diff'] < 0.05:
        print(f"      ✅ Minimal overfitting (excellent generalization)")
    elif result['overfit_diff'] < 0.10:
        print(f"      ✅ Good generalization")
    elif result['overfit_diff'] < 0.15:
        print(f"      ⚠️  Moderate overfitting")
    else:
        print(f"      ❌ Significant overfitting")

    print(f"\n   💡 Interpretation:")
    print(f"      Mean salary: ${result['mean_salary']:,.0f}")
    print(f"      MAE is {result['mae_pct']:.1f}% of mean salary")
    print(f"      Model explains {result['test_r2']*100:.1f}% of salary variance")
    print(f"      Training time: {result['train_time']:.2f} seconds")


def print_comparison_table(results):
    """Print comparison table of all combinations."""
    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON SUMMARY - NLP METHODS WITH SBERT PCA")
    print("="*120)

    # Sort by test R² (descending)
    sorted_results = sorted(results, key=lambda x: x["test_r2"], reverse=True)

    print(f"\n{'NLP Method':<12} {'Model':<35} {'Features':<10} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<12} {'Overfit':<10} {'Time(s)':<10}")
    print("-"*120)

    for r in sorted_results:
        print(
            f"{r['nlp_method']:<12} "
            f"{r['model_name']:<35} "
            f"{r['n_features']:<10} "
            f"{r['test_r2']:<10.4f} "
            f"${r['test_rmse']:<11,.0f} "
            f"${r['test_mae']:<11,.0f} "
            f"{r['overfit_diff']:<10.4f} "
            f"{r['train_time']:<10.2f}"
        )

    # Best overall
    best = sorted_results[0]
    print(f"\n🏆 Best Overall: {best['nlp_method'].upper()} + {best['model_name']}")
    print(f"   Test R²:  {best['test_r2']:.4f}")
    print(f"   Test RMSE: ${best['test_rmse']:,.0f}")
    print(f"   Test MAE:  ${best['test_mae']:,.0f}")
    print(f"   Features:  {best['n_features']:,}")
    print(f"   Training time: {best['train_time']:.2f} seconds")
    print(f"   Overfitting: {best['overfit_diff']:.4f}")

    # Best by NLP method
    print(f"\n📊 Best Model for Each NLP Method:")
    for nlp_method in NLP_METHODS:
        method_results = [r for r in sorted_results if r['nlp_method'] == nlp_method]
        if method_results:
            best_method = max(method_results, key=lambda x: x["test_r2"])
            print(f"   {nlp_method.upper():<12} → {best_method['model_name']:<35} (R²: {best_method['test_r2']:.4f}, Overfit: {best_method['overfit_diff']:.4f})")

    # Best by model type
    print(f"\n📊 Best NLP Method for Each Model:")
    model_types_used = sorted(set(r['model_type'] for r in sorted_results))
    for model_type in model_types_used:
        model_results = [r for r in sorted_results if r['model_type'] == model_type]
        if model_results:
            best_model = max(model_results, key=lambda x: x["test_r2"])
            print(f"   {get_model_name(model_type):<35} → {best_model['nlp_method'].upper():<12} (R²: {best_model['test_r2']:.4f}, Overfit: {best_model['overfit_diff']:.4f})")

    # Overfitting analysis
    print(f"\n📈 Overfitting Analysis:")
    excellent = [r for r in sorted_results if r['overfit_diff'] < 0.05]
    good = [r for r in sorted_results if 0.05 <= r['overfit_diff'] < 0.10]
    moderate = [r for r in sorted_results if 0.10 <= r['overfit_diff'] < 0.15]
    significant = [r for r in sorted_results if r['overfit_diff'] >= 0.15]

    print(f"   ✅ Excellent generalization (overfit < 0.05): {len(excellent)} models")
    print(f"   ✅ Good generalization (overfit 0.05-0.10): {len(good)} models")
    print(f"   ⚠️  Moderate overfitting (overfit 0.10-0.15): {len(moderate)} models")
    print(f"   ❌ Significant overfitting (overfit ≥ 0.15): {len(significant)} models")


def main():
    print("="*120)
    print("NLP METHOD & MODEL COMPARISON - SBERT PCA FEATURES")
    print("="*120)
    print(f"\nComparing:")
    print(f"   NLP Methods: {', '.join([m.upper() for m in NLP_METHODS])}")
    print(f"   Models: {', '.join([get_model_name(m) for m in MODELS_TO_TRAIN])}")
    print(f"   Total combinations: {len(NLP_METHODS) * len(MODELS_TO_TRAIN)}")
    print(f"   Input directory: {CONFIG['input_dir']}/")

    # Check available models
    available_models = []
    for model_type in MODELS_TO_TRAIN:
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            print(f"   ⚠️  Skipping XGBoost (not available)")
            continue
        if model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            print(f"   ⚠️  Skipping LightGBM (not available)")
            continue
        if model_type == "catboost" and not CATBOOST_AVAILABLE:
            print(f"   ⚠️  Skipping CatBoost (not available)")
            continue
        available_models.append(model_type)

    print(f"\n   Available models: {len(available_models)}")
    print(f"   Expected total combinations: {len(NLP_METHODS) * len(available_models)}")

    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load original data to get target variable
    print(f"\n{'='*120}")
    print("LOADING ORIGINAL DATA FOR TARGET VARIABLE")
    print(f"{'='*120}")

    # Load y (target) from the original file
    print(f"\n📖 Loading target variable from salary_data_ready_for_new_nlp.csv...")
    with open('salary_data_ready_for_new_nlp.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        y_data = []
        for row in reader:
            y_data.append(float(row['salary_normalized']))

    y = np.array(y_data)
    print(f"   ✓ Loaded {len(y):,} target values")
    print(f"   Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"   Salary mean: ${y.mean():,.0f}")

    # Train models for each NLP method
    all_results = []

    for nlp_method in NLP_METHODS:
        # Load feature file
        feature_file = os.path.join(CONFIG["input_dir"], f"salary_data_nlp_{nlp_method}.csv")

        if not os.path.exists(feature_file):
            print(f"\n❌ ERROR: File not found: {feature_file}")
            continue

        print(f"\n{'='*120}")
        print(f"PROCESSING NLP METHOD: {nlp_method.upper()}")
        print(f"{'='*120}")

        # Load features
        X, feature_names = load_feature_file(feature_file)

        # Preprocess: train/test split
        X_train, X_test = preprocess_data(X, nlp_method)

        # Split target variable the same way
        y_train, y_test = train_test_split(
            y,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state']
        )

        print(f"\n✓ Data ready for training:")
        print(f"   Features: {X_train.shape[1]:,}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")

        # Train each model
        for model_type in available_models:
            model_name = get_model_name(model_type)
            print(f"\n{'─'*120}")
            print(f"Training: {nlp_method.upper()} + {model_name}")
            print(f"{'─'*120}")

            result = train_model(X_train, y_train, X_test, y_test, nlp_method, model_type)

            if result:
                print_detailed_results(result)
                all_results.append(result)

                # Save model (optional)
                model_file = os.path.join(
                    CONFIG["output_dir"],
                    f"model_{nlp_method}_{model_type}.pkl"
                )
                save_data = {
                    "model": result["model"],
                    "scaler": result["scaler"],
                    "feature_names": feature_names,
                }
                with open(model_file, "wb") as f:
                    pickle.dump(save_data, f)
                print(f"\n   💾 Saved model to {model_file}")
            else:
                print(f"   ⚠️  Skipped {model_name} for {nlp_method.upper()}")

    # Print comprehensive comparison
    if all_results:
        print_comparison_table(all_results)

        # Save comparison results
        comparison_info = {
            "best_overall": max(all_results, key=lambda x: x["test_r2"]),
            "all_results": all_results,
            "summary": {
                "total_combinations": len(all_results),
                "nlp_methods": NLP_METHODS,
                "models": available_models,
                "config": CONFIG,
            }
        }

        info_file = os.path.join(CONFIG["output_dir"], "comparison_results.pkl")
        with open(info_file, "wb") as f:
            pickle.dump(comparison_info, f)
        print(f"\n💾 Saved comparison results to {info_file}")

        print("\n" + "="*120)
        print("✅ COMPREHENSIVE COMPARISON COMPLETE!")
        print("="*120)
    else:
        print("\n❌ ERROR: No models were trained successfully!")


if __name__ == "__main__":
    main()
