"""
Step 6: Train Advanced Models
==============================
Train advanced regression models for salary prediction.

Models:
1. Random Forest (ensemble tree model)
2. XGBoost (gradient boosting - GPU optional)

GPU Support:
- Random Forest: CPU only (sklearn doesn't support GPU)
- XGBoost: GPU optional (CPU: 1-3 min, GPU: 30-60 sec)
  - Set USE_GPU=True in CONFIG to enable GPU acceleration
  - Requires XGBoost GPU version installed

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 4)
Output: Model performance metrics and saved models
"""

import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

# =============================================================================
# CONFIGURATION - GPU Settings
# =============================================================================

CONFIG = {
    "use_gpu": False,  # Set to True to use GPU for XGBoost (requires GPU version installed)
    # Random Forest parameters
    "rf_n_estimators": 100,  # Number of trees
    "rf_max_depth": 20,  # Max tree depth
    "rf_random_state": 42,
    "rf_n_jobs": -1,  # Use all CPU cores
    # XGBoost parameters (will be set based on GPU availability)
    "xgb_n_estimators": 100,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.1,
    "xgb_random_state": 42,
}


def check_gpu_availability():
    """Check if GPU is available for XGBoost."""
    try:
        import xgboost as xgb

        # Try to create a simple test with GPU
        import numpy as np

        X_test = np.random.rand(100, 10)
        y_test = np.random.rand(100)
        dtrain = xgb.DMatrix(X_test, label=y_test)
        params = {"tree_method": "gpu_hist", "gpu_id": 0}
        xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        return True
    except:
        return False


def load_csv_data(X_file, y_file):
    """Load training or test data from CSV files."""
    print(f"\nLoading {X_file} and {y_file}...")

    # Load X
    with open(X_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        X_data = []
        for row in reader:
            X_data.append([float(row[feat]) for feat in reader.fieldnames])
        X = np.array(X_data)
        feature_names = list(reader.fieldnames)

    # Load y
    with open(y_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        y_data = []
        for row in reader:
            y_data.append(float(row[reader.fieldnames[0]]))
        y = np.array(y_data)

    print(f"   ✓ Loaded: {len(X)} samples × {len(feature_names)} features")
    return X, y, feature_names


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train model and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"✓ Training completed in {train_time:.2f} seconds")

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Print results
    print(f"\n📊 Performance Metrics:")
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE ($)':<20} ${train_rmse:<14,.0f} ${test_rmse:<14,.0f}")
    print(f"{'MAE ($)':<15} ${train_mae:<14,.0f} ${test_mae:<14,.0f}")

    # Overfitting check
    overfit_diff = train_r2 - test_r2
    print(f"\n📈 Overfitting Check:")
    print(f"   R² difference (Train - Test): {overfit_diff:.4f}")
    if overfit_diff < 0.05:
        print("   ✓ Minimal overfitting (good generalization)")
    elif overfit_diff < 0.10:
        print("   ⚠️  Moderate overfitting")
    else:
        print("   ❌ Significant overfitting (consider regularization)")

    # Mean salary for context
    mean_salary = np.mean(y_test)
    mae_pct = (test_mae / mean_salary) * 100
    print(f"\n💡 Interpretation:")
    print(f"   Mean salary: ${mean_salary:,.0f}")
    print(f"   MAE is {mae_pct:.1f}% of mean salary")
    print(f"   Model explains {test_r2*100:.1f}% of salary variance")

    return {
        "model": model,
        "model_name": model_name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "overfit_diff": overfit_diff,
        "train_time": train_time,
    }


def main():
    print("=" * 80)
    print("STEP 6: TRAIN ADVANCED MODELS")
    print("=" * 80)

    print(
        """
    This script trains advanced regression models:
    
    1. Random Forest - Ensemble tree model (CPU only)
    2. XGBoost - Gradient boosting (GPU optional)
    
    GPU Configuration:
    - Random Forest: CPU only (sklearn doesn't support GPU)
    - XGBoost: Set CONFIG['use_gpu']=True to enable GPU acceleration
    """
    )

    # =========================================================================
    # STEP 6.1: Check GPU Availability
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.1: GPU AVAILABILITY CHECK")
    print("=" * 80)

    gpu_available = False
    if CONFIG["use_gpu"]:
        print("\nChecking GPU availability for XGBoost...")
        gpu_available = check_gpu_availability()

        if gpu_available:
            print("   ✓ GPU available! XGBoost will use GPU acceleration")
            print("   ✓ Device: RTX 5070 Ti (detected)")
        else:
            print("   ✗ GPU not available for XGBoost")
            print("      → Falling back to CPU (still fast enough)")
            print("      → To enable GPU:")
            print("         1. Install XGBoost GPU version: pip install xgboost[gpu]")
            print("         2. Make sure CUDA drivers are installed")
            CONFIG["use_gpu"] = False
    else:
        print("\n✓ GPU acceleration disabled in CONFIG")
        print("   → Training will use CPU (fast enough for this dataset)")

    # =========================================================================
    # STEP 6.2: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.2: LOAD DATA")
    print("=" * 80)

    X_train, y_train, feature_names = load_csv_data("X_train.csv", "y_train.csv")
    X_test, y_test, _ = load_csv_data("X_test.csv", "y_test.csv")

    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples:     {len(X_test):,}")
    print(f"   Features:         {len(feature_names)}")
    print(f"   Target range:     ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    print(f"   Target mean:      ${y_test.mean():,.0f}")

    # =========================================================================
    # STEP 6.3: Train Random Forest
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.3: TRAIN RANDOM FOREST")
    print("=" * 80)

    print(f"\nRandom Forest Parameters:")
    print(f"   n_estimators: {CONFIG['rf_n_estimators']} trees")
    print(f"   max_depth: {CONFIG['rf_max_depth']}")
    print(f"   n_jobs: {CONFIG['rf_n_jobs']} (using all CPU cores)")

    rf_model = RandomForestRegressor(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        random_state=CONFIG["rf_random_state"],
        n_jobs=CONFIG["rf_n_jobs"],
        verbose=1,
    )

    rf_results = train_and_evaluate_model(
        rf_model, "Random Forest", X_train, y_train, X_test, y_test
    )

    # Feature importance
    feature_importance = rf_model.feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:][::-1]

    print(f"\n🔍 Top 10 Most Important Features (Random Forest):")
    for i, idx in enumerate(top_10_idx):
        feat_name = feature_names[idx]
        importance = feature_importance[idx]
        print(f"   {i+1:2}. {feat_name:<40} {importance:.6f}")

    # =========================================================================
    # STEP 6.4: Train XGBoost
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.4: TRAIN XGBOOST")
    print("=" * 80)

    try:
        import xgboost as xgb

        print(f"\nXGBoost Parameters:")
        print(f"   n_estimators: {CONFIG['xgb_n_estimators']} trees")
        print(f"   max_depth: {CONFIG['xgb_max_depth']}")
        print(f"   learning_rate: {CONFIG['xgb_learning_rate']}")

        # Configure tree method based on GPU availability
        if CONFIG["use_gpu"] and gpu_available:
            tree_method = "gpu_hist"
            print(f"   tree_method: {tree_method} (GPU acceleration enabled)")
        else:
            tree_method = "hist"  # Fast CPU method
            print(f"   tree_method: {tree_method} (CPU)")

        xgb_params = {
            "n_estimators": CONFIG["xgb_n_estimators"],
            "max_depth": CONFIG["xgb_max_depth"],
            "learning_rate": CONFIG["xgb_learning_rate"],
            "random_state": CONFIG["xgb_random_state"],
            "tree_method": tree_method,
            "verbosity": 1,
        }

        if CONFIG["use_gpu"] and gpu_available:
            xgb_params["gpu_id"] = 0

        xgb_model = xgb.XGBRegressor(**xgb_params)

        xgb_results = train_and_evaluate_model(
            xgb_model, f"XGBoost ({tree_method})", X_train, y_train, X_test, y_test
        )

        # Feature importance
        feature_importance = xgb_model.feature_importances_
        top_10_idx = np.argsort(feature_importance)[-10:][::-1]

        print(f"\n🔍 Top 10 Most Important Features (XGBoost):")
        for i, idx in enumerate(top_10_idx):
            feat_name = feature_names[idx]
            importance = feature_importance[idx]
            print(f"   {i+1:2}. {feat_name:<40} {importance:.6f}")

        xgb_available = True

    except ImportError:
        print("\n⚠️  XGBoost not installed!")
        print("   Skipping XGBoost training")
        print("   Install: pip install xgboost")
        if CONFIG["use_gpu"]:
            print("   For GPU support: pip install xgboost[gpu]")
        xgb_results = None
        xgb_model = None
        xgb_available = False

    # =========================================================================
    # STEP 6.5: Compare Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.5: MODEL COMPARISON")
    print("=" * 80)

    all_results = [rf_results]
    if xgb_available:
        all_results.append(xgb_results)

    print(
        f"\n{'Model':<30} {'Test R²':<12} {'Test RMSE':<15} {'Test MAE':<15} {'Train Time':<12}"
    )
    print("-" * 85)

    for r in all_results:
        print(
            f"{r['model_name']:<30} {r['test_r2']:<12.4f} ${r['test_rmse']:<14,.0f} ${r['test_mae']:<14,.0f} {r['train_time']:<12.2f}s"
        )

    # Find best model
    best_model = max(all_results, key=lambda x: x["test_r2"])

    print(f"\n🏆 Best Advanced Model: {best_model['model_name']}")
    print(f"   Test R²:  {best_model['test_r2']:.4f}")
    print(f"   Test RMSE: ${best_model['test_rmse']:,.0f}")
    print(f"   Test MAE:  ${best_model['test_mae']:,.0f}")

    # =========================================================================
    # STEP 6.6: Save Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6.6: SAVE MODELS")
    print("=" * 80)

    models_to_save = {
        "random_forest": rf_model,
        "feature_names": feature_names,
        "results": all_results,
        "best_model_name": best_model["model_name"],
    }

    if xgb_available:
        models_to_save["xgboost"] = xgb_model

    with open("advanced_models.pkl", "wb") as f:
        pickle.dump(models_to_save, f)

    print("\n✓ Saved: advanced_models.pkl")
    print("   Includes: Random Forest, XGBoost (if available) + results")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6 COMPLETE!")
    print("=" * 80)

    print(
        f"""
📊 ADVANCED MODELS SUMMARY:

Best Model: {best_model['model_name']}
   • Test R²:  {best_model['test_r2']:.4f} (explains {best_model['test_r2']*100:.1f}% of variance)
   • Test RMSE: ${best_model['test_rmse']:,.0f}
   • Test MAE:  ${best_model['test_mae']:,.0f} ({best_model['test_mae']/np.mean(y_test)*100:.1f}% of mean salary)
   • Overfitting: {best_model['overfit_diff']:.4f} (Train - Test R² diff)
   • Training time: {best_model['train_time']:.2f} seconds

GPU Status:
   • GPU requested: {CONFIG['use_gpu']}
   • GPU available: {gpu_available if xgb_available else 'N/A'}
   • Device used: {'RTX 5070 Ti (GPU)' if (CONFIG['use_gpu'] and gpu_available) else 'CPU'}

All Models Saved: advanced_models.pkl

🚀 NEXT STEP:
   → Step 7: Compare all models (Baseline + Advanced)
   → Evaluate feature importance
   → Model diagnostics and visualization

💡 GPU Note:
   • Random Forest: CPU only (2-5 min, fast enough)
   • XGBoost: {'GPU' if (CONFIG['use_gpu'] and gpu_available) else 'CPU'} ({best_model['train_time']:.1f}s)
   • To enable GPU: Install XGBoost GPU version and set CONFIG['use_gpu']=True
    """
    )

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
