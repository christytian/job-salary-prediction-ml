"""
Baseline Models Without NLP Features
=====================================
Trains Linear Regression, Random Forest, XGBoost, and MLP models using ONLY
original numeric and one-hot encoded features (NO NLP features).

This establishes a baseline to measure the value added by NLP features.
"""

import csv
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")


def load_data(filename):
    """Load data from CSV file"""
    print(f"\nLoading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        all_fieldnames = list(reader.fieldnames)

    print(f"   Loaded: {len(data):,} records, {len(all_fieldnames)} columns")
    return data, all_fieldnames


def identify_nlp_features(fieldnames):
    """Identify NLP features to drop"""
    nlp_features = []

    for col in fieldnames:
        # TF-IDF features
        if 'tfidf' in col.lower():
            nlp_features.append(col)
        # SBERT embedding features
        elif 'emb_' in col:
            nlp_features.append(col)
        # Word2Vec embedding features
        elif '_w2v_' in col:
            nlp_features.append(col)

    return nlp_features


def prepare_features(data, all_fieldnames):
    """Prepare features by dropping NLP features"""

    # Target variable
    target_col = 'salary_normalized'

    # Identify NLP features to drop
    nlp_features = identify_nlp_features(all_fieldnames)

    print(f"\n Identified {len(nlp_features)} NLP features to DROP:")
    tfidf_count = sum(1 for f in nlp_features if 'tfidf' in f.lower())
    sbert_count = sum(1 for f in nlp_features if 'emb_' in f)
    w2v_count = sum(1 for f in nlp_features if '_w2v_' in f)
    print(f"   - {tfidf_count} TF-IDF features")
    print(f"   - {sbert_count} SBERT embedding features")
    print(f"   - {w2v_count} Word2Vec embedding features")

    # Columns to exclude (target + NLP + non-numeric)
    exclude_cols = set(nlp_features)
    exclude_cols.add(target_col)
    exclude_cols.add('normalized_salary')  # Data leakage
    exclude_cols.add('salary')
    exclude_cols.add('min_salary')
    exclude_cols.add('max_salary')
    exclude_cols.add('salary_min')
    exclude_cols.add('salary_max')

    # Also exclude text columns
    text_cols = ['title', 'description', 'company_name', 'location',
                 'description_company', 'title_cleaned', 'description_cleaned',
                 'all_skills_cleaned', 'benefits_list', 'all_industries',
                 'all_skills', 'all_company_industries', 'all_specialities']
    exclude_cols.update(text_cols)

    # Get numeric feature columns (excluding NLP)
    feature_cols = []
    for col in all_fieldnames:
        if col in exclude_cols:
            continue

        # Sample to check if numeric
        sample_val = None
        for row in data[:100]:
            if row.get(col) and row[col] != '':
                sample_val = row[col]
                break

        if sample_val:
            try:
                float(sample_val)
                feature_cols.append(col)
            except ValueError:
                pass

    print(f"\n Keeping {len(feature_cols)} features (NO NLP):")

    # Categorize remaining features
    onehot_features = [col for col in feature_cols if '_' in col and
                       any(prefix in col for prefix in ['work_type_', 'exp_level_',
                                                         'currency_', 'country_', 'state_',
                                                         'skill_', 'industry_', 'speciality_'])]
    missingness_indicators = [col for col in feature_cols if col.endswith('_missing')]
    original_numeric = [col for col in feature_cols
                       if col not in onehot_features
                       and col not in missingness_indicators]

    print(f"   - {len(original_numeric)} original numeric features")
    print(f"   - {len(onehot_features)} one-hot encoded features")
    print(f"   - {len(missingness_indicators)} missingness indicators")

    # Extract features and target
    X = []
    y = []

    for row in data:
        features = [float(row[col]) if row[col] else 0.0 for col in feature_cols]
        target = float(row[target_col])
        X.append(features)
        y.append(target)

    return np.array(X), np.array(y), feature_cols


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, needs_scaling=False):
    """Train model and return performance metrics"""
    print(f"\n{'-' * 80}")
    print(f"Training {model_name}...")
    print(f"{'-' * 80}")

    start_time = time.time()

    # Scale if needed
    if needs_scaling:
        print("   Applying StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    overfitting = train_r2 - test_r2

    print(f"   Training time: {train_time:.2f}s")
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Test RMSE: ${test_rmse:,.0f}")
    print(f"   Test MAE:  ${test_mae:,.0f}")
    print(f"   Overfitting: {overfitting:.4f} ({'Low' if overfitting < 0.05 else 'Moderate' if overfitting < 0.1 else 'High'})")

    return {
        'model': model,
        'name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfitting': overfitting,
        'train_time': train_time
    }


def main():
    print("=" * 80)
    print("BASELINE MODELS WITHOUT NLP FEATURES")
    print("=" * 80)
    print("\nPurpose: Establish baseline performance using ONLY:")
    print("   • Original numeric features")
    print("   • One-hot encoded categorical features")
    print("   • Missingness indicators")
    print("\nNO NLP features (TF-IDF or embeddings) are used!")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    try:
        data, all_fieldnames = load_data('salary_data_with_nlp_hybrid_onehot_features.csv')
    except FileNotFoundError:
        print("\n ERROR: salary_data_with_nlp_hybrid_onehot_features.csv not found!")
        return

    # =========================================================================
    # STEP 2: Prepare Features (Drop NLP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPARE FEATURES (DROP ALL NLP)")
    print("=" * 80)

    X, y, feature_names = prepare_features(data, all_fieldnames)

    print(f"\n Final Dataset:")
    print(f"   Records: {len(X):,}")
    print(f"   Features: {len(feature_names)} (NO NLP)")
    print(f"   Target: salary_normalized")
    print(f"   Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"   Salary mean: ${y.mean():,.0f}")
    print(f"   Salary std: ${y.std():,.0f}")

    # =========================================================================
    # STEP 3: Train/Test Split
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN/TEST SPLIT (80/20)")
    print("=" * 80)

    print("\nSplitting data with random_state=42 for reproducibility...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test set:     {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    # =========================================================================
    # STEP 4: Train Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN BASELINE MODELS")
    print("=" * 80)

    results = []

    # 1. Linear Regression
    print("\n" + "=" * 80)
    print("MODEL 1: Linear Regression")
    print("=" * 80)
    lr = LinearRegression()
    results.append(train_and_evaluate(
        lr, X_train, X_test, y_train, y_test,
        "Linear Regression", needs_scaling=True
    ))

    # 2. Random Forest
    print("\n" + "=" * 80)
    print("MODEL 2: Random Forest Regressor")
    print("=" * 80)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    results.append(train_and_evaluate(
        rf, X_train, X_test, y_train, y_test,
        "Random Forest", needs_scaling=False
    ))

    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 80)
        print("MODEL 3: XGBoost Regressor")
        print("=" * 80)
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        results.append(train_and_evaluate(
            xgb, X_train, X_test, y_train, y_test,
            "XGBoost", needs_scaling=False
        ))

    # 4. MLP Neural Network
    print("\n" + "=" * 80)
    print("MODEL 4: MLP Neural Network")
    print("=" * 80)
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    results.append(train_and_evaluate(
        mlp, X_train, X_test, y_train, y_test,
        "MLP Neural Network", needs_scaling=True
    ))

    # =========================================================================
    # STEP 5: Compare Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: BASELINE COMPARISON (NO NLP)")
    print("=" * 80)

    # Sort by test R² (descending)
    results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)

    print("\nPerformance Summary (sorted by Test R²):")
    print("\n" + "-" * 120)
    print(f"{'Rank':<6} {'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<15} {'Test MAE':<15} {'Overfitting':<12} {'Time (s)':<10}")
    print("-" * 120)

    for rank, result in enumerate(results_sorted, 1):
        print(f"{rank:<6} {result['name']:<25} "
              f"{result['train_r2']:<12.4f} "
              f"{result['test_r2']:<12.4f} "
              f"${result['test_rmse']:<14,.0f} "
              f"${result['test_mae']:<14,.0f} "
              f"{result['overfitting']:<12.4f} "
              f"{result['train_time']:<10.2f}")

    print("-" * 120)

    # =========================================================================
    # STEP 6: Best Baseline Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: BEST BASELINE MODEL (NO NLP)")
    print("=" * 80)

    best = results_sorted[0]

    print(f"\n BEST BASELINE: {best['name']}")
    print(f"\n Performance (WITHOUT NLP features):")
    print(f"   Test R²:  {best['test_r2']:.4f} (explains {best['test_r2']*100:.2f}% of variance)")
    print(f"   Test RMSE: ${best['test_rmse']:,.0f}")
    print(f"   Test MAE:  ${best['test_mae']:,.0f} ({best['test_mae']/y.mean()*100:.1f}% of avg salary)")
    print(f"   Overfitting: {best['overfitting']:.4f}")
    print(f"   Training time: {best['train_time']:.2f}s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)

    print(f"""
 Baseline Performance WITHOUT NLP Features:

 Best Model: {best['name']}
   • Test R² = {best['test_r2']:.4f}
   • Test RMSE = ${best['test_rmse']:,.0f}
   • Test MAE = ${best['test_mae']:,.0f}

 Features Used ({len(feature_names)} total):
   • Original numeric features (continuous)
   • One-hot encoded categorical features
   • Missingness indicators
   • NO TF-IDF features
   • NO Embedding features

 Next Step:
   Compare these baseline results against models WITH NLP features
   to measure the value added by TF-IDF and embeddings!
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
