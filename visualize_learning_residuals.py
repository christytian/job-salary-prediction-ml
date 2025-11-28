"""
Learning Curves and Residual Analysis for Baseline Models
==========================================================
Creates diagnostic visualizations for baseline models (no NLP features):
1. Learning curves (4 models) - shows performance vs training set size
2. Residual analysis (4 models) - diagnostic plots for model assumptions
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy import stats
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


def plot_learning_curves(models_dict, X, y, output_file='baseline_learning_curves.png'):
    """Create learning curves for all models"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Learning Curves: Baseline Models (No NLP Features)',
                 fontsize=16, fontweight='bold', y=0.995)

    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']

    for idx, (ax, model_name) in enumerate(zip(axes.flat, model_names)):
        print(f"\n[{idx+1}/4] Generating learning curve for {model_name}...")

        model = models_dict[model_name]

        # Define training sizes (10%, 20%, ..., 100%)
        train_sizes = np.linspace(0.1, 1.0, 10)

        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,  # 5-fold cross-validation
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#2ecc71',
                linewidth=2, markersize=6, label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.2, color='#2ecc71')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='#e74c3c',
                linewidth=2, markersize=6, label='Cross-Validation Score')
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.2, color='#e74c3c')

        # Final performance
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val

        ax.set_title(f'{model_name}\nFinal: Train R²={final_train:.4f}, CV R²={final_val:.4f}, Gap={gap:.4f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Set Size (samples)', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add convergence line
        ax.axhline(y=final_val, color='gray', linestyle='--',
                  linewidth=1, alpha=0.5, label=f'CV Plateau: {final_val:.4f}')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved learning curves to: {output_file}")
    plt.close()


def plot_residual_analysis(models_dict, predictions_dict, y_test, output_file='baseline_residual_analysis.png'):
    """Create residual analysis plots for all models"""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Residual Analysis: Baseline Models (No NLP Features)',
                 fontsize=16, fontweight='bold', y=0.995)

    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP Neural Network']

    for idx, model_name in enumerate(model_names):
        print(f"\n[{idx+1}/4] Generating residual plots for {model_name}...")

        y_pred = predictions_dict[model_name]
        residuals = y_test - y_pred

        # Row for this model (3 plots per row)
        row = idx

        # Plot 1: Residuals vs Predicted
        ax1 = plt.subplot(4, 3, row*3 + 1)
        ax1.scatter(y_pred, residuals, alpha=0.3, s=20, edgecolors='none')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Salary ($)', fontsize=10)
        ax1.set_ylabel('Residuals ($)', fontsize=10)
        ax1.set_title(f'{model_name}\nResiduals vs Predicted', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Format axis
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Plot 2: Residual Distribution (Histogram)
        ax2 = plt.subplot(4, 3, row*3 + 2)
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals ($)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'Residual Distribution\nMean={residuals.mean():,.0f}, Std={residuals.std():,.0f}',
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Format axis
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Plot 3: Q-Q Plot (Normal Probability Plot)
        ax3 = plt.subplot(4, 3, row*3 + 3)
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title(f'Q-Q Plot\n(Check Normality of Residuals)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax3.set_ylabel('Sample Quantiles', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Get the line to format y-axis
        line = ax3.get_lines()[0]
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved residual analysis to: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("LEARNING CURVES AND RESIDUAL ANALYSIS")
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

    # Prepare models for learning curves
    print("\n" + "=" * 80)
    print("STEP 4: GENERATE LEARNING CURVES")
    print("=" * 80)

    # Create model instances (use simpler models for faster learning curve computation)
    models_for_learning = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        'MLP Neural Network': MLPRegressor(
            hidden_layer_sizes=(128, 64),  # Smaller for speed
            max_iter=100,  # Reduced for speed
            random_state=42,
            verbose=False
        )
    }

    # Generate learning curves
    plot_learning_curves(models_for_learning, X, y, 'baseline_learning_curves.png')

    # Train full models for residual analysis
    print("\n" + "=" * 80)
    print("STEP 5: TRAIN MODELS FOR RESIDUAL ANALYSIS")
    print("=" * 80)

    predictions = {}
    models_full = {}

    # 1. Linear Regression
    print("\n[1/4] Training Linear Regression...")
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_test_scaled = scaler_lr.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    predictions['Linear Regression'] = lr.predict(X_test_scaled)
    models_full['Linear Regression'] = lr
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
    models_full['Random Forest'] = rf
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
    models_full['XGBoost'] = xgb
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
    models_full['MLP Neural Network'] = mlp
    print("   ✓ Complete")

    # Generate residual analysis
    print("\n" + "=" * 80)
    print("STEP 6: GENERATE RESIDUAL ANALYSIS")
    print("=" * 80)

    plot_residual_analysis(models_full, predictions, y_test, 'baseline_residual_analysis.png')

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)

    print(f"""
✓ Generated 2 comprehensive diagnostic visualizations:

  1. baseline_learning_curves.png (16x14 inches, 300 DPI)
     • Shows how model performance improves with more training data
     • Training score vs Cross-Validation score
     • Helps diagnose:
       - Underfitting: Both curves low and close together
       - Overfitting: Large gap between train and CV curves
       - Convergence: Whether more data would help
     • Shaded bands show score variance (±1 std)

  2. baseline_residual_analysis.png (20x16 inches, 300 DPI)
     • 3 diagnostic plots per model (12 plots total):

       a) Residuals vs Predicted:
          - Should show random scatter around zero
          - Patterns indicate model issues
          - Funnel shape = heteroscedasticity (non-constant variance)

       b) Residual Distribution (Histogram):
          - Should be approximately normal (bell curve)
          - Centered at zero
          - Shows mean and std of residuals

       c) Q-Q Plot (Normality Check):
          - Points should follow diagonal line
          - Deviations indicate non-normal residuals
          - Important for statistical inference

Key Insights to Look For:
  • Learning curves: Do models benefit from more data?
  • Residuals: Are model assumptions met?
  • Patterns in residuals = systematic errors to fix
  • Normal residuals = good model fit

Ready for publication, presentations, or model diagnostics!
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
