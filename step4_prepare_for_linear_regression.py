"""
Step 4: Prepare Dataset for Linear Regression
==============================================
Validates critical fields, drops non-numeric columns, and creates a clean dataset
ready for linear regression modeling.
"""

import csv

# =============================================================================
# CONFIGURATION - Critical fields that MUST exist for modeling
# =============================================================================

CRITICAL_FIELDS = {
    'title': {
        'required': True,
        'min_length': 3,
        'purpose': 'Primary text for NLP features (TF-IDF and embeddings)'
    },
    'description': {
        'required': True,
        'min_length': 20,
        'purpose': 'Primary text for NLP features (TF-IDF and embeddings)'
    },
    'salary_normalized': {
        'required': True,
        'min_value': 10000,
        'max_value': 1000000,
        'purpose': 'Target variable for regression'
    }
}

# Non-numeric columns to drop (text/categorical - already encoded in NLP features)
NON_NUMERIC_COLUMNS_TO_DROP = [
    # Text columns - already encoded in TF-IDF and embeddings
    'title',
    'description',
    'company_name',
    'location',
    'description_company',

    # Categorical columns - would need one-hot encoding, but not needed for baseline
    'formatted_work_type',
    'formatted_experience_level',
    'currency',
    'country',
    'state',
    'city',
    'primary_skill',
    'primary_industry',
    'all_industries',
    'all_skills',
    'primary_company_industry',
    'all_company_industries',
    'primary_speciality',
    'all_specialities',
    'benefits_list',

    # Cleaned text versions (used for NLP, not needed as features)
    'title_cleaned',
    'description_cleaned',
    'all_skills_cleaned'
]


def validate_critical_fields(data, critical_fields):
    """
    Validate critical fields and return clean data + validation report

    Returns:
        clean_data: List of valid rows
        invalid_rows: List of (index, row, reasons) for invalid rows
    """
    clean_data = []
    invalid_rows = []

    for idx, row in enumerate(data):
        reasons = []

        for field, rules in critical_fields.items():
            value = row.get(field, '').strip()

            # Check if required field is missing/empty
            if rules.get('required') and not value:
                reasons.append(f"{field}: MISSING or EMPTY")
                continue

            # Check minimum length for text fields
            if 'min_length' in rules and len(value) < rules['min_length']:
                reasons.append(f"{field}: Too short ({len(value)} chars, min {rules['min_length']})")

            # Check numeric ranges
            if 'min_value' in rules or 'max_value' in rules:
                try:
                    num_val = float(value)
                    if 'min_value' in rules and num_val < rules['min_value']:
                        reasons.append(f"{field}: ${num_val:,.0f} < min ${rules['min_value']:,}")
                    if 'max_value' in rules and num_val > rules['max_value']:
                        reasons.append(f"{field}: ${num_val:,.0f} > max ${rules['max_value']:,}")
                except ValueError:
                    reasons.append(f"{field}: Not a valid number")

        if reasons:
            invalid_rows.append((idx, row, reasons))
        else:
            clean_data.append(row)

    return clean_data, invalid_rows


def main():
    print("=" * 80)
    print("STEP 4: PREPARE DATASET FOR LINEAR REGRESSION")
    print("=" * 80)

    # =========================================================================
    # STEP 4.1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.1: LOAD DATA")
    print("=" * 80)

    print("\nLoading salary_data_with_nlp_features.csv...")
    try:
        with open('salary_data_with_nlp_features.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            all_fieldnames = list(reader.fieldnames)
        print(f" Loaded: {len(data):,} records, {len(all_fieldnames)} features")
    except FileNotFoundError:
        print(" ERROR: salary_data_with_nlp_features.csv not found!")
        print("   Please run step3_nlp_hybrid.py first to generate this file.")
        return

    # =========================================================================
    # STEP 4.2: Validate Critical Fields
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.2: VALIDATE CRITICAL FIELDS")
    print("=" * 80)

    print("\nValidating critical fields (title, description, salary_normalized)...")
    print("\nValidation rules:")
    for field, rules in CRITICAL_FIELDS.items():
        print(f"\n   {field}:")
        print(f"      Required: {rules['required']}")
        if 'min_length' in rules:
            print(f"      Min length: {rules['min_length']} characters")
        if 'min_value' in rules:
            print(f"      Min value: ${rules['min_value']:,}")
        if 'max_value' in rules:
            print(f"      Max value: ${rules['max_value']:,}")
        print(f"      Purpose: {rules['purpose']}")

    original_count = len(data)
    clean_data, invalid_rows = validate_critical_fields(data, CRITICAL_FIELDS)

    if invalid_rows:
        print(f"\n  VALIDATION FAILED for {len(invalid_rows)} rows:")
        for idx, row, reasons in invalid_rows:
            print(f"\n   Row {idx} (CSV row {idx + 2}):")
            print(f"      Title: {row.get('title', 'N/A')[:60]}")
            print(f"      Company: {row.get('company_name', 'N/A')[:40]}")
            print(f"      Salary: {row.get('salary_normalized', 'N/A')}")
            print(f"      Issues:")
            for reason in reasons:
                print(f"         • {reason}")

        # Save invalid rows to file for inspection
        invalid_file = 'step4_invalid_rows.csv'
        with open(invalid_file, 'w', encoding='utf-8', newline='') as f:
            if invalid_rows:
                fieldnames = list(all_fieldnames) + ['validation_errors']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx, row, reasons in invalid_rows:
                    row['validation_errors'] = ' | '.join(reasons)
                    writer.writerow(row)
        print(f"\n    Saved invalid rows to: {invalid_file}")

        print(f"\n Removed {len(invalid_rows)} invalid rows ({len(invalid_rows)/original_count*100:.3f}% of data)")
        print(f" Continuing with {len(clean_data):,} valid rows")
    else:
        print("\n NO VALIDATION ISSUES FOUND - All rows passed validation!")

    data = clean_data

    # =========================================================================
    # STEP 4.3: Identify Numeric vs Non-Numeric Columns
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.3: IDENTIFY NUMERIC VS NON-NUMERIC COLUMNS")
    print("=" * 80)

    target_col = 'salary_normalized'

    # Columns to exclude (target + data leakage columns)
    exclude_cols = {
        target_col,
        'normalized_salary',  # Data leakage - same as target
        'salary',             # Original salary field
        'min_salary',         # Salary range fields
        'max_salary',
        'salary_min',
        'salary_max'
    }

    numeric_cols = []
    non_numeric_cols = []

    # Check each column (except target and leakage columns)
    for col in all_fieldnames:
        if col in exclude_cols:
            continue

        # Sample first non-empty value to check type
        sample_val = None
        for row in data[:100]:
            if row.get(col) and row[col] != '':
                sample_val = row[col]
                break

        if sample_val:
            try:
                float(sample_val)
                numeric_cols.append(col)
            except ValueError:
                non_numeric_cols.append(col)

    print(f"\n Column Type Summary:")
    print(f"   Numeric columns:     {len(numeric_cols):>4}")
    print(f"   Non-numeric columns: {len(non_numeric_cols):>4}")
    print(f"   Target variable:     {1:>4} ({target_col})")

    # =========================================================================
    # STEP 4.4: Drop Non-Numeric Columns
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.4: DROP NON-NUMERIC COLUMNS")
    print("=" * 80)

    print(f"\n Dropping {len(non_numeric_cols)} non-numeric columns:")
    print("\n   These columns are text/categorical and already encoded in NLP features:")

    # Categorize for better output
    text_cols = [col for col in non_numeric_cols if col in [
        'title', 'description', 'company_name', 'location', 'description_company',
        'title_cleaned', 'description_cleaned', 'all_skills_cleaned'
    ]]
    categorical_cols = [col for col in non_numeric_cols if col not in text_cols]

    if text_cols:
        print(f"\n   Text columns ({len(text_cols)}):")
        for col in text_cols:
            print(f"      • {col}")

    if categorical_cols:
        print(f"\n   Categorical columns ({len(categorical_cols)}):")
        for col in categorical_cols:
            print(f"      • {col}")

    print("\n    Their information is preserved in:")
    print("      • 199 TF-IDF features (keyword-based)")
    print("      • 768 Embedding features (semantic meaning)")

    # =========================================================================
    # STEP 4.5: Build Final Feature Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.5: BUILD FINAL FEATURE SET (NUMERIC ONLY)")
    print("=" * 80)

    # Final fieldnames = numeric columns + target
    final_fieldnames = numeric_cols + [target_col]

    # Categorize features for reporting
    tfidf_features = [col for col in numeric_cols if 'tfidf' in col.lower()]
    embedding_features = [col for col in numeric_cols if 'emb_' in col]
    missingness_indicators = [col for col in numeric_cols if col.endswith('_missing')]
    original_numeric = [col for col in numeric_cols
                       if col not in tfidf_features
                       and col not in embedding_features
                       and col not in missingness_indicators]

    print(f"\n Final Feature Breakdown:")
    print(f"   TF-IDF features:            {len(tfidf_features):>4}")
    print(f"   Embedding features:         {len(embedding_features):>4}")
    print(f"   Original numeric features:  {len(original_numeric):>4}")
    print(f"   Missingness indicators:     {len(missingness_indicators):>4}")
    print(f"   {'─' * 40}")
    print(f"   Total features:             {len(numeric_cols):>4}")
    print(f"   Target variable:            {1:>4}")
    print(f"   {'─' * 40}")
    print(f"   Total columns:              {len(final_fieldnames):>4}")

    print(f"\n Sample Features:")
    if tfidf_features:
        print(f"\n   TF-IDF (first 5): {', '.join(tfidf_features[:5])}")
    if embedding_features:
        print(f"\n   Embeddings (first 5): {', '.join(embedding_features[:5])}")
    if original_numeric:
        print(f"\n   Original Numeric: {', '.join(original_numeric[:10])}")
    if missingness_indicators:
        print(f"\n   Missingness Indicators: {', '.join(missingness_indicators)}")

    # =========================================================================
    # STEP 4.6: Save Clean Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.6: SAVE CLEAN DATASET FOR LINEAR REGRESSION")
    print("=" * 80)

    output_file = 'salary_data_lr_ready.csv'
    print(f"\nSaving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    print(f" Saved: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" DATASET IS READY FOR LINEAR REGRESSION!")
    print("=" * 80)

    print(f"\n Final Dataset:")
    print(f"   File:     {output_file}")
    print(f"   Records:  {len(data):,} ({original_count - len(data)} rows dropped)")
    print(f"   Features: {len(numeric_cols)} (all numeric)")
    print(f"   Target:   {target_col} (continuous, numeric)")

    print(f"\n What's Included:")
    print(f"    {len(tfidf_features)} TF-IDF features (keyword-based)")
    print(f"    {len(embedding_features)} Embedding features (semantic)")
    print(f"    {len(original_numeric)} Original numeric features")
    print(f"    {len(missingness_indicators)} Missingness indicators")
    print(f"    All features are numeric - ready for linear regression!")

    print(f"\n  What Was Removed:")
    print(f"    {len(non_numeric_cols)} text/categorical columns")
    print(f"      (Information preserved in TF-IDF/embeddings)")
    print(f"    {original_count - len(data)} rows with missing/invalid data")
    print(f"      ({(original_count - len(data))/original_count*100:.3f}% of data - negligible loss)")

    print(f"\n Data Quality Metrics:")
    print(f"   • No missing values in critical fields ")
    print(f"   • All salaries in valid range ($10K - $1M) ")
    print(f"   • All features are numeric ")
    print(f"   • Ready for train/test split ")

    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 80)

    print(f"""
  Train/Test Split (80/20):
   from sklearn.model_selection import train_test_split
   import pandas as pd

   df = pd.read_csv('{output_file}')
   X = df.drop('salary_normalized', axis=1)
   y = df['salary_normalized']
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

  Feature Scaling (StandardScaler):
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

  Train Linear Regression:
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X_train_scaled, y_train)

  Evaluate:
   from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

   y_pred = model.predict(X_test_scaled)
   rmse = mean_squared_error(y_test, y_pred, squared=False)
   r2 = r2_score(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)

   print(f"RMSE: ${{rmse:,.0f}}")
   print(f"R²:   {{r2:.4f}}")
   print(f"MAE:  ${{mae:,.0f}}")

  Check for Multicollinearity (Optional):
   from statsmodels.stats.outliers_influence import variance_inflation_factor

   vif_data = pd.DataFrame()
   vif_data["feature"] = X.columns
   vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                      for i in range(len(X.columns))]
   print(vif_data.sort_values('VIF', ascending=False).head(20))
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
