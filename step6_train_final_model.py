"""
Step 5: Train and Evaluate Linear Regression Model
===================================================
Trains a baseline linear regression model on the salary prediction dataset
and evaluates performance with comprehensive metrics.
"""

import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def load_data(filename):
    """Load CSV data into numpy arrays"""
    print(f"\nLoading {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = list(reader.fieldnames)

    print(f" Loaded: {len(data):,} records, {len(fieldnames)} columns")

    return data, fieldnames


def prepare_features_target(data, fieldnames, target_col='salary_normalized'):
    """Convert data to numpy arrays for X and y"""
    feature_cols = [col for col in fieldnames if col != target_col]

    print(f"\nConverting to numpy arrays...")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target:   {target_col}")

    # Extract features (X) and target (y)
    X = []
    y = []

    for row in data:
        # Features
        feature_values = []
        for col in feature_cols:
            val = row[col]
            try:
                feature_values.append(float(val))
            except (ValueError, TypeError):
                feature_values.append(0.0)  # Should not happen with clean data
        X.append(feature_values)

        # Target
        try:
            y.append(float(row[target_col]))
        except (ValueError, TypeError):
            y.append(0.0)  # Should not happen with clean data

    X = np.array(X)
    y = np.array(y)

    print(f" X shape: {X.shape}")
    print(f" y shape: {y.shape}")

    return X, y, feature_cols


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Residuals
    residuals = y_true - y_pred

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'mape': mape,
        'residuals': residuals
    }


def analyze_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Analyze prediction errors in detail"""
    errors = np.abs(y_true - y_pred)

    # Sort errors
    sorted_indices = np.argsort(errors)

    print(f"\n {dataset_name} Prediction Analysis:")
    print(f"   {'─' * 50}")

    # Percentiles of errors
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n   Absolute Error Percentiles:")
    for p in percentiles:
        val = np.percentile(errors, p)
        print(f"      {p:2d}th: ${val:>10,.0f}")

    # Best predictions (smallest errors)
    print(f"\n    Best 5 Predictions (Smallest Errors):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"      Actual: ${y_true[idx]:>8,.0f}  |  Predicted: ${y_pred[idx]:>8,.0f}  |  Error: ${errors[idx]:>6,.0f}")

    # Worst predictions (largest errors)
    print(f"\n    Worst 5 Predictions (Largest Errors):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        print(f"      Actual: ${y_true[idx]:>8,.0f}  |  Predicted: ${y_pred[idx]:>8,.0f}  |  Error: ${errors[idx]:>6,.0f}")

    # Error distribution
    print(f"\n    Error Distribution:")
    ranges = [
        (0, 5000, "< $5K"),
        (5000, 10000, "$5K - $10K"),
        (10000, 20000, "$10K - $20K"),
        (20000, 50000, "$20K - $50K"),
        (50000, float('inf'), "> $50K")
    ]

    for low, high, label in ranges:
        count = np.sum((errors >= low) & (errors < high))
        pct = count / len(errors) * 100
        print(f"      {label:<15} {count:>6,} ({pct:>5.1f}%)")


def main():
    print("=" * 80)
    print("STEP 5: TRAIN AND EVALUATE LINEAR REGRESSION MODEL")
    print("=" * 80)

    # =========================================================================
    # STEP 5.1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.1: LOAD DATA")
    print("=" * 80)

    try:
        data, fieldnames = load_data('salary_data_lr_ready.csv')
    except FileNotFoundError:
        print(" ERROR: salary_data_lr_ready.csv not found!")
        print("   Please run step4_prepare_for_linear_regression.py first.")
        return

    # =========================================================================
    # STEP 5.2: Prepare Features and Target
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.2: PREPARE FEATURES AND TARGET")
    print("=" * 80)

    X, y, feature_cols = prepare_features_target(data, fieldnames)

    # Quick stats on target
    print(f"\n Target Variable Statistics:")
    print(f"   Min:    ${y.min():>10,.0f}")
    print(f"   Max:    ${y.max():>10,.0f}")
    print(f"   Mean:   ${y.mean():>10,.0f}")
    print(f"   Median: ${np.median(y):>10,.0f}")
    print(f"   Std:    ${y.std():>10,.0f}")

    # =========================================================================
    # STEP 5.3: Train/Test Split
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.3: TRAIN/TEST SPLIT (80/20)")
    print("=" * 80)

    print("\nSplitting data with random_state=42 for reproducibility...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n Split complete:")
    print(f"   Training set:   {X_train.shape[0]:>6,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Test set:       {X_test.shape[0]:>6,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"   Features:       {X_train.shape[1]:>6,}")

    # =========================================================================
    # STEP 5.4: Feature Scaling
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.4: FEATURE SCALING (StandardScaler)")
    print("=" * 80)

    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f" Scaling complete")
    print(f"\n   Features are now standardized:")
    print(f"      Mean ≈ 0, Std ≈ 1")

    # =========================================================================
    # STEP 5.5: Train Linear Regression Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.5: TRAIN LINEAR REGRESSION MODEL")
    print("=" * 80)

    print("\nInitializing Linear Regression model...")
    model = LinearRegression()

    print("Training model on scaled training data...")
    print(f"   (This may take a moment with {X_train.shape[1]} features...)")

    model.fit(X_train_scaled, y_train)

    print(f"\n Model trained successfully!")
    print(f"   Coefficients: {model.coef_.shape[0]:,}")
    print(f"   Intercept:    ${model.intercept_:,.0f}")

    # =========================================================================
    # STEP 5.6: Evaluate Model - Training Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.6: EVALUATE MODEL - TRAINING SET")
    print("=" * 80)

    print("\nMaking predictions on training set...")
    y_train_pred = model.predict(X_train_scaled)

    train_metrics = calculate_metrics(y_train, y_train_pred)

    print(f"\n Training Set Performance:")
    print(f"   {'─' * 50}")
    print(f"   R² Score:  {train_metrics['r2']:>8.4f}  (1.0 = perfect fit)")
    print(f"   RMSE:      ${train_metrics['rmse']:>10,.0f}")
    print(f"   MAE:       ${train_metrics['mae']:>10,.0f}")
    print(f"   MAPE:      {train_metrics['mape']:>8.2f}%")

    analyze_predictions(y_train, y_train_pred, "Training Set")

    # =========================================================================
    # STEP 5.7: Evaluate Model - Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.7: EVALUATE MODEL - TEST SET")
    print("=" * 80)

    print("\nMaking predictions on test set...")
    y_test_pred = model.predict(X_test_scaled)

    test_metrics = calculate_metrics(y_test, y_test_pred)

    print(f"\n Test Set Performance:")
    print(f"   {'─' * 50}")
    print(f"   R² Score:  {test_metrics['r2']:>8.4f}  (1.0 = perfect fit)")
    print(f"   RMSE:      ${test_metrics['rmse']:>10,.0f}")
    print(f"   MAE:       ${test_metrics['mae']:>10,.0f}")
    print(f"   MAPE:      {test_metrics['mape']:>8.2f}%")

    analyze_predictions(y_test, y_test_pred, "Test Set")

    # =========================================================================
    # STEP 5.8: Compare Train vs Test Performance
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.8: TRAIN VS TEST PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<15} {'Training':>12} {'Test':>12} {'Difference':>12}")
    print(f"   {'─' * 55}")
    print(f"{'R² Score':<15} {train_metrics['r2']:>12.4f} {test_metrics['r2']:>12.4f} {train_metrics['r2'] - test_metrics['r2']:>12.4f}")
    print(f"{'RMSE':<15} ${train_metrics['rmse']:>11,.0f} ${test_metrics['rmse']:>11,.0f} ${train_metrics['rmse'] - test_metrics['rmse']:>11,.0f}")
    print(f"{'MAE':<15} ${train_metrics['mae']:>11,.0f} ${test_metrics['mae']:>11,.0f} ${train_metrics['mae'] - test_metrics['mae']:>11,.0f}")
    print(f"{'MAPE':<15} {train_metrics['mape']:>11.2f}% {test_metrics['mape']:>11.2f}% {train_metrics['mape'] - test_metrics['mape']:>11.2f}%")

    # Check for overfitting
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    print(f"\n Overfitting Analysis:")
    if r2_diff < 0.01:
        print(f"    EXCELLENT - No overfitting detected (R² diff: {r2_diff:.4f})")
    elif r2_diff < 0.05:
        print(f"    GOOD - Minimal overfitting (R² diff: {r2_diff:.4f})")
    elif r2_diff < 0.10:
        print(f"     MODERATE - Some overfitting (R² diff: {r2_diff:.4f})")
    else:
        print(f"    SIGNIFICANT - High overfitting (R² diff: {r2_diff:.4f})")

    # =========================================================================
    # STEP 5.9: Feature Importance (Top Coefficients)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5.9: FEATURE IMPORTANCE (TOP COEFFICIENTS)")
    print("=" * 80)

    # Get absolute coefficients and sort
    coef_abs = np.abs(model.coef_)
    top_indices = np.argsort(coef_abs)[::-1]

    print(f"\n Top 20 Most Important Features (by absolute coefficient):")
    print(f"   {'Rank':<6} {'Feature':<50} {'Coefficient':>15}")
    print(f"   {'─' * 75}")

    for rank, idx in enumerate(top_indices[:20], 1):
        feature_name = feature_cols[idx]
        coef_val = model.coef_[idx]
        print(f"   {rank:<6} {feature_name:<50} {coef_val:>15,.2f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" LINEAR REGRESSION MODEL SUMMARY")
    print("=" * 80)

    print(f"\n Final Model Performance:")
    print(f"   Dataset:           salary_data_lr_ready.csv")
    print(f"   Total samples:     {len(X):,}")
    print(f"   Training samples:  {len(X_train):,}")
    print(f"   Test samples:      {len(X_test):,}")
    print(f"   Features:          {X.shape[1]:,}")

    print(f"\n Test Set Metrics (What matters most):")
    print(f"   R² Score:          {test_metrics['r2']:.4f}")
    print(f"   RMSE:              ${test_metrics['rmse']:,.0f}")
    print(f"   MAE:               ${test_metrics['mae']:,.0f}")
    print(f"   MAPE:              {test_metrics['mape']:.2f}%")

    print(f"\n Interpretation:")
    if test_metrics['r2'] >= 0.7:
        print(f"    EXCELLENT - Model explains {test_metrics['r2']*100:.1f}% of salary variance")
    elif test_metrics['r2'] >= 0.5:
        print(f"    GOOD - Model explains {test_metrics['r2']*100:.1f}% of salary variance")
    elif test_metrics['r2'] >= 0.3:
        print(f"     MODERATE - Model explains {test_metrics['r2']*100:.1f}% of salary variance")
    else:
        print(f"    POOR - Model explains only {test_metrics['r2']*100:.1f}% of salary variance")

    avg_salary = y_test.mean()
    error_pct = (test_metrics['mae'] / avg_salary) * 100
    print(f"\n   Average test error: ${test_metrics['mae']:,.0f} ({error_pct:.1f}% of avg salary)")

    # =========================================================================
    # NEXT STEPS & RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 80)

    print("""
  Model Improvements:
   • Try Ridge/Lasso regression for regularization
   • Feature selection to reduce dimensionality
   • Polynomial features for non-linear relationships
   • Ensemble methods (Random Forest, Gradient Boosting)

  Feature Engineering:
   • Feature interactions (e.g., experience × seniority)
   • Dimensionality reduction (PCA on embeddings)
   • Target transformation (log scale for salaries)

  Diagnostics:
   • Check for multicollinearity (VIF analysis)
   • Residual analysis (plot residuals vs predicted)
   • Learning curves (to diagnose bias vs variance)

  Model Deployment:
   • Save model using joblib/pickle
   • Create prediction API/interface
   • Monitor model performance over time
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
