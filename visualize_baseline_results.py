"""
Visualize Baseline Model Results
=================================
Creates visualizations for baseline models (no NLP features):
1. Actual vs Predicted scatter plots for all 4 models
2. Feature importance charts for Random Forest and XGBoost
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


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

    # Extract features and target
    X = []
    y = []

    for row in data:
        features = [float(row[col]) if row[col] else 0.0 for col in feature_cols]
        target = float(row[target_col])
        X.append(features)
        y.append(target)

    return np.array(X), np.array(y), feature_cols


def plot_actual_vs_predicted(y_test, predictions_dict, output_file='baseline_scatter_plots.png'):
    """Create 2x2 grid of actual vs predicted scatter plots"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Baseline Models: Actual vs Predicted Salary (No NLP Features)',
                 fontsize=16, fontweight='bold', y=0.995)

    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']

    for idx, (ax, model_name) in enumerate(zip(axes.flat, models)):
        y_pred = predictions_dict[model_name]

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.3, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Title with metrics
        ax.set_title(f'{model_name}\nR² = {r2:.4f} | RMSE = ${rmse:,.0f} | MAE = ${mae:,.0f}',
                    fontsize=12, fontweight='bold')

        ax.set_xlabel('Actual Salary ($)', fontsize=11)
        ax.set_ylabel('Predicted Salary ($)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format axis labels as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved scatter plots to: {output_file}")
    plt.close()


def plot_feature_importance(rf_model, xgb_model, feature_names, output_file='baseline_feature_importance.png'):
    """Create side-by-side feature importance plots for RF and XGBoost"""

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Feature Importance: Baseline Models (27 Features, No NLP)',
                 fontsize=16, fontweight='bold')

    # Random Forest importance
    rf_importance = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importance)[::-1][:20]  # Top 20

    ax = axes[0]
    ax.barh(range(len(rf_indices)), rf_importance[rf_indices], color='#2ecc71')
    ax.set_yticks(range(len(rf_indices)))
    ax.set_yticklabels([feature_names[i] for i in rf_indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest\nTop 20 Most Important Features',
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add importance values on bars
    for i, idx in enumerate(rf_indices):
        ax.text(rf_importance[idx] + 0.001, i, f'{rf_importance[idx]:.4f}',
               va='center', fontsize=9)

    # XGBoost importance
    xgb_importance = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importance)[::-1][:20]  # Top 20

    ax = axes[1]
    ax.barh(range(len(xgb_indices)), xgb_importance[xgb_indices], color='#3498db')
    ax.set_yticks(range(len(xgb_indices)))
    ax.set_yticklabels([feature_names[i] for i in xgb_indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost\nTop 20 Most Important Features',
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add importance values on bars
    for i, idx in enumerate(xgb_indices):
        ax.text(xgb_importance[idx] + 0.001, i, f'{xgb_importance[idx]:.4f}',
               va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance to: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("VISUALIZE BASELINE MODEL RESULTS")
    print("=" * 80)

    # Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    try:
        data, all_fieldnames = load_data('salary_data_with_nlp_hybrid_onehot_features.csv')
    except FileNotFoundError:
        print("\n ERROR: salary_data_with_nlp_hybrid_onehot_features.csv not found!")
        return

    # Prepare features
    print("\n" + "=" * 80)
    print("STEP 2: PREPARE FEATURES (DROP NLP)")
    print("=" * 80)

    X, y, feature_names = prepare_features(data, all_fieldnames)
    print(f"\n✓ Dataset ready: {len(X):,} records, {len(feature_names)} features")

    # Train/test split
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train models and get predictions
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN MODELS AND GENERATE PREDICTIONS")
    print("=" * 80)

    predictions = {}

    # 1. Linear Regression
    print("\n[1/4] Training Linear Regression...")
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_test_scaled = scaler_lr.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    predictions['Linear Regression'] = lr.predict(X_test_scaled)
    print("   ✓ Complete")

    # 2. Random Forest
    print("\n[2/4] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    predictions['Random Forest'] = rf.predict(X_test)
    print("   ✓ Complete")

    # 3. XGBoost
    print("\n[3/4] Training XGBoost...")
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
    xgb.fit(X_train, y_train)
    predictions['XGBoost'] = xgb.predict(X_test)
    print("   ✓ Complete")

    # 4. MLP Neural Network
    print("\n[4/4] Training MLP Neural Network...")
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
    mlp.fit(X_train_scaled, y_train)
    predictions['MLP Neural Network'] = mlp.predict(X_test_scaled)
    print("   ✓ Complete")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("STEP 5: GENERATE VISUALIZATIONS")
    print("=" * 80)

    print("\n[1/2] Creating Actual vs Predicted scatter plots...")
    plot_actual_vs_predicted(y_test, predictions, 'baseline_scatter_plots.png')

    print("\n[2/2] Creating Feature Importance plots...")
    plot_feature_importance(rf, xgb, feature_names, 'baseline_feature_importance.png')

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)

    print(f"""
✓ Generated 2 visualization files:

  1. baseline_scatter_plots.png (16x14 inches, 300 DPI)
     • 2x2 grid of Actual vs Predicted scatter plots
     • All 4 models: Linear Regression, Random Forest, XGBoost, MLP
     • Shows R², RMSE, MAE for each model
     • Perfect prediction diagonal reference line

  2. baseline_feature_importance.png (18x10 inches, 300 DPI)
     • Side-by-side feature importance comparison
     • Random Forest vs XGBoost
     • Top 20 most important features (out of 27 baseline features)
     • Shows which non-NLP features drive salary predictions

Key Insights:
  • Scatter plots reveal prediction patterns and errors
  • Feature importance shows which baseline features matter most
  • Ready for publication or presentation!
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
