"""
Generate NLP PCA Comparison Summary Table
==========================================
Create a formatted table similar to the reference showing:
- Model name
- Train R²
- Test R²
- Overfitting Gap
- Overfit Status
"""

import pickle
import pandas as pd

# Load comparison results
with open('models_nlp_pca_comparison/comparison_results.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# Get all results
results = results_dict['all_results']

# Prepare data for table
table_data = []

for result in results:
    nlp_method = result['nlp_method'].upper()
    model_name = result['model_name']
    train_r2 = result['train_r2']
    test_r2 = result['test_r2']
    overfit_gap = result['overfit_diff']

    # Determine overfit status
    if overfit_gap < 0.05:
        status = "None"
    elif overfit_gap < 0.10:
        status = "Better"
    elif overfit_gap < 0.15:
        status = "Moderate"
    else:
        status = "Severe"

    table_data.append({
        'NLP Method': nlp_method,
        'Model': model_name,
        'Train R²': f"{train_r2:.4f}",
        'Test R²': f"{test_r2:.4f}",
        'Overfitting Gap': f"{overfit_gap:.4f}",
        'Overfit Status': status
    })

# Create DataFrame
df = pd.DataFrame(table_data)

# Sort by Test R² descending
df_sorted = df.sort_values('Test R²', ascending=False)

print("=" * 120)
print("NLP PCA COMPARISON - MODEL PERFORMANCE SUMMARY")
print("=" * 120)
print()
print(df_sorted.to_string(index=False))
print()

# Group by NLP method
print("=" * 120)
print("GROUPED BY NLP METHOD")
print("=" * 120)

for method in ['TFIDF', 'SBERT_PCA', 'HYBRID']:
    print(f"\n{method}:")
    print("-" * 120)
    df_method = df[df['NLP Method'] == method].sort_values('Test R²', ascending=False)
    print(df_method[['Model', 'Train R²', 'Test R²', 'Overfitting Gap', 'Overfit Status']].to_string(index=False))

# Summary statistics
print("\n" + "=" * 120)
print("SUMMARY STATISTICS BY NLP METHOD")
print("=" * 120)

for method in ['TFIDF', 'SBERT_PCA', 'HYBRID']:
    df_method = df[df['NLP Method'] == method]
    test_r2_values = [float(x) for x in df_method['Test R²']]
    overfit_values = [float(x) for x in df_method['Overfitting Gap']]

    print(f"\n{method}:")
    print(f"  Average Test R²: {sum(test_r2_values)/len(test_r2_values):.4f}")
    print(f"  Best Test R²: {max(test_r2_values):.4f}")
    print(f"  Worst Test R²: {min(test_r2_values):.4f}")
    print(f"  Average Overfitting: {sum(overfit_values)/len(overfit_values):.4f}")
    print(f"  Min Overfitting: {min(overfit_values):.4f}")
    print(f"  Max Overfitting: {max(overfit_values):.4f}")

print("\n" + "=" * 120)
