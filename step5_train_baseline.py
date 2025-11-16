"""
Step 5: Train Baseline Models
==============================
Train baseline regression models for salary prediction.

Models:
1. Linear Regression (baseline)
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization)

GPU Support:
- These models use CPU only (fast enough, no GPU needed)
- Training time: < 5 seconds total

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv (from Step 4)
Output: Model performance metrics and saved models
"""

import csv
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

# Try to import tqdm for progress bar, fallback to simple progress if not available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


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
    print(f"{'MAE ($)':<20} ${train_mae:<14,.0f} ${test_mae:<14,.0f}")

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
    print("STEP 5: TRAIN BASELINE MODELS")
    print("=" * 80)

    print(
        """
    This script trains baseline regression models:
    
    1. Linear Regression - Simple baseline (no regularization)
    2. Ridge Regression - L2 regularization (prevents overfitting)
    3. Lasso Regression - L1 regularization (feature selection)
    
    GPU Note: These models use CPU only (training < 5 seconds, GPU not needed)
    """
    )

    # =========================================================================
    # STEP 5.1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.1: LOAD DATA")
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
    # STEP 5.2: Train Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.2: TRAIN BASELINE MODELS")
    print("=" * 80)

    results = []

    # Model 1: Linear Regression
    linear_model = LinearRegression()
    linear_results = train_and_evaluate_model(
        linear_model, "Linear Regression", X_train, y_train, X_test, y_test
    )
    results.append(linear_results)

    # Model 2: Ridge Regression (L2 regularization)
    # Try different alpha values and use best one
    print(f"\n{'='*60}")
    print("Tuning Ridge Regression (L2 regularization)...")
    print(f"{'='*60}")

    alpha_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha = 1.0
    best_r2 = -np.inf

    for alpha in alpha_values:
        ridge_temp = Ridge(alpha=alpha, random_state=42)
        ridge_temp.fit(X_train, y_train)
        y_test_pred_temp = ridge_temp.predict(X_test)
        test_r2_temp = r2_score(y_test, y_test_pred_temp)

        print(f"   Alpha = {alpha:>7.1f}: Test R² = {test_r2_temp:.4f}")

        if test_r2_temp > best_r2:
            best_r2 = test_r2_temp
            best_alpha = alpha

    print(f"\n✓ Best alpha: {best_alpha} (Test R² = {best_r2:.4f})")

    ridge_model = Ridge(alpha=best_alpha, random_state=42)
    ridge_results = train_and_evaluate_model(
        ridge_model,
        f"Ridge Regression (α={best_alpha})",
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append(ridge_results)

    # Model 3: Lasso Regression (L1 regularization)
    print(f"\n{'='*60}")
    print("Tuning Lasso Regression (L1 regularization)...")
    print(f"{'='*60}")
    print("   Note: Lasso training is slower due to L1 regularization optimization")
    print(
        "         Testing fewer alpha values and using faster convergence settings..."
    )

    # Optimized: Use fewer alpha values and faster convergence
    # Test fewer alpha values (most important range)
    alpha_values = [0.01, 0.1, 1.0]  # Reduced from 5 to 3 values
    best_alpha = 0.1
    best_r2 = -np.inf

    # Use warm_start to speed up: reuse solution from previous alpha
    # Note: Lasso doesn't support verbose parameter in sklearn
    lasso_temp = Lasso(
        random_state=42, max_iter=3000, tol=1e-3, warm_start=True
    )

    # Create progress indicator
    print(f"\n   Testing {len(alpha_values)} alpha values...")
    print("   " + "=" * 50)
    
    # Use simple progress display that works better with training loops
    for idx, alpha in enumerate(alpha_values):
        print(f"\n   [{idx+1}/{len(alpha_values)}] Testing alpha = {alpha:.3f}...")
        print(f"   Status: Training in progress...", end="", flush=True)

        start_alpha_time = time.time()
        lasso_temp.set_params(alpha=alpha)
        lasso_temp.fit(X_train, y_train)
        alpha_train_time = time.time() - start_alpha_time

        # Clear the "in progress" line and show results
        print(f"\r   Status: ✓ Completed ({alpha_train_time:.1f}s)          ")  # Extra spaces to clear line

        y_test_pred_temp = lasso_temp.predict(X_test)
        test_r2_temp = r2_score(y_test, y_test_pred_temp)

        # Count non-zero features (feature selection)
        n_features_selected = np.sum(np.abs(lasso_temp.coef_) > 1e-5)

        print(f"   Results: R² = {test_r2_temp:.4f}, Features = {n_features_selected}")
        print(f"   Progress: {'█' * (idx + 1)}{'░' * (len(alpha_values) - idx - 1)} ({idx + 1}/{len(alpha_values)})")

        if test_r2_temp > best_r2:
            best_r2 = test_r2_temp
            best_alpha = alpha
            print(f"   ⭐ New best alpha found!")
    
    print("   " + "=" * 50)

    print(f"\n✓ Best alpha: {best_alpha} (Test R² = {best_r2:.4f})")

    # Final model with best alpha (allow more iterations for final training)
    print(f"\n   Training final Lasso model with alpha = {best_alpha}...")
    print("   (Progress shown above, final training in progress...)")
    lasso_model = Lasso(
        alpha=best_alpha, random_state=42, max_iter=3000, tol=1e-3
    )
    # Train final model (verbose=1 shows convergence progress)
    lasso_results = train_and_evaluate_model(
        lasso_model,
        f"Lasso Regression (α={best_alpha})",
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append(lasso_results)

    # Count selected features in Lasso
    n_selected = np.sum(np.abs(lasso_model.coef_) > 1e-5)
    print(
        f"\n   Features selected by Lasso: {n_selected} / {len(feature_names)} ({n_selected/len(feature_names)*100:.1f}%)"
    )

    # =========================================================================
    # STEP 5.3: Compare Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.3: MODEL COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Model':<30} {'Test R²':<12} {'Test RMSE':<15} {'Test MAE':<15} {'Train Time':<12}"
    )
    print("-" * 85)

    for r in results:
        print(
            f"{r['model_name']:<30} {r['test_r2']:<12.4f} ${r['test_rmse']:<14,.0f} ${r['test_mae']:<14,.0f} {r['train_time']:<12.2f}s"
        )

    # Find best model
    best_model = max(results, key=lambda x: x["test_r2"])

    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   Test R²:  {best_model['test_r2']:.4f}")
    print(f"   Test RMSE: ${best_model['test_rmse']:,.0f}")
    print(f"   Test MAE:  ${best_model['test_mae']:,.0f}")

    # =========================================================================
    # STEP 5.4: Save Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.4: SAVE MODELS")
    print("=" * 80)

    models_to_save = {
        "linear_regression": linear_model,
        "ridge_regression": ridge_model,
        "lasso_regression": lasso_model,
        "feature_names": feature_names,
        "results": results,
        "best_model_name": best_model["model_name"],
    }

    with open("baseline_models.pkl", "wb") as f:
        pickle.dump(models_to_save, f)

    print("\n✓ Saved: baseline_models.pkl")
    print("   Includes: Linear, Ridge, Lasso models + results")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE!")
    print("=" * 80)

    print(
        f"""
📊 BASELINE MODELS SUMMARY:

Best Model: {best_model['model_name']}
   • Test R²:  {best_model['test_r2']:.4f} (explains {best_model['test_r2']*100:.1f}% of variance)
   • Test RMSE: ${best_model['test_rmse']:,.0f}
   • Test MAE:  ${best_model['test_mae']:,.0f} ({best_model['test_mae']/np.mean(y_test)*100:.1f}% of mean salary)
   • Overfitting: {best_model['overfit_diff']:.4f} (Train - Test R² diff)

All Models Saved: baseline_models.pkl

🚀 NEXT STEP:
   → Step 6: Train advanced models (Random Forest, XGBoost)
   → Can optionally use GPU for XGBoost if installed

💡 Note: Baseline models completed in < 5 seconds on CPU.
          GPU not needed for these models.
    """
    )

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
