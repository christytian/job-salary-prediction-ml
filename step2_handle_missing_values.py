"""
Feature Engineering Step 2: Handle Missing Values
==================================================
Drop high-missing features and impute remaining missing values
"""

import csv
from collections import Counter


def get_mode(values):
    """Get the most common value from a list."""
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def get_median(values):
    """Get median from a list of numeric values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    else:
        return sorted_vals[n//2]


def main():
    print("=" * 80)
    print("STEP 2: HANDLE MISSING VALUES (IMPUTE + INDICATOR APPROACH)")
    print("=" * 80)

    # Load data
    print("\nLoading salary_data_final.csv...")
    with open('salary_data_final.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        all_fields = list(reader.fieldnames)

    print(f"✓ Loaded: {len(data):,} records, {len(all_fields)} features")

    # =========================================================================
    # STEP 2.0: Analyze Missing Value Tiers
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.0: ANALYZE MISSING VALUE TIERS")
    print("=" * 80)

    missing_analysis = {}
    for col in all_fields:
        missing_count = sum(1 for row in data if not row.get(col) or row[col] == '')
        if missing_count > 0:
            pct = missing_count / len(data) * 100
            missing_analysis[col] = {'count': missing_count, 'pct': pct}

    # Categorize into tiers
    tier_1_drop = []      # ≥90% missing
    tier_2_indicator = []  # 30-90% missing (impute + add indicator)
    tier_3_impute = []     # <30% missing (impute only)

    for col, info in missing_analysis.items():
        pct = info['pct']
        if pct >= 90:
            tier_1_drop.append(col)
        elif pct >= 30:
            tier_2_indicator.append(col)
        else:
            tier_3_impute.append(col)

    print(f"\n📊 Missing Value Tier Breakdown:")
    print(f"   Tier 1 (≥90% missing - DROP):              {len(tier_1_drop)} features")
    print(f"   Tier 2 (30-90% missing - IMPUTE+INDICATOR): {len(tier_2_indicator)} features")
    print(f"   Tier 3 (<30% missing - IMPUTE ONLY):        {len(tier_3_impute)} features")

    # =========================================================================
    # STEP 2.1: Drop Tier 1 Features (≥90% missing)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.1: DROP TIER 1 FEATURES (≥90% MISSING)")
    print("=" * 80)

    # Also drop redundant salary columns (now that we have salary_normalized)
    redundant_features = [
        'salary_yearly_min',
        'salary_yearly_max',
        'salary_yearly_med',
        'min_salary',
        'max_salary',
        'med_salary'
    ]

    features_to_drop = list(set(tier_1_drop + redundant_features))

    print(f"\nDropping {len(features_to_drop)} features:")
    print("\n   Tier 1 (≥90% missing):")
    for f in tier_1_drop:
        if f in all_fields:
            pct = missing_analysis.get(f, {}).get('pct', 0)
            print(f"      ❌ {f:<30} ({pct:.1f}% missing)")

    if redundant_features:
        print("\n   Redundant salary columns:")
        for f in redundant_features:
            if f in all_fields:
                print(f"      ❌ {f:<30} (redundant with salary_normalized)")

    # Keep only features not in drop list
    features_to_keep = [f for f in all_fields if f not in features_to_drop]

    print(f"\n✓ Kept: {len(features_to_keep)} features")
    print(f"✓ Dropped: {len(features_to_drop)} features")

    # =========================================================================
    # STEP 2.2: Create Missingness Indicators for Tier 2 Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.2: CREATE MISSINGNESS INDICATORS FOR TIER 2 FEATURES")
    print("=" * 80)

    # Filter tier_2_indicator to only include features we're keeping
    tier_2_to_process = [col for col in tier_2_indicator if col in features_to_keep]

    indicator_columns = []
    indicator_counts = {}

    if tier_2_to_process:
        print(f"\nCreating '{col}_missing' binary indicators for {len(tier_2_to_process)} Tier 2 features:")
        print("(These preserve the signal that a value was originally missing)\n")

        for col in tier_2_to_process:
            indicator_col = f"{col}_missing"
            indicator_columns.append(indicator_col)

            missing_count = 0
            for row in data:
                is_missing = not row.get(col) or row[col] == ''
                row[indicator_col] = '1' if is_missing else '0'
                if is_missing:
                    missing_count += 1

            indicator_counts[col] = missing_count
            pct = missing_count / len(data) * 100
            print(f"   ✓ {indicator_col:<35} (captures {missing_count:,} missing values, {pct:.1f}%)")

        print(f"\n✓ Created {len(indicator_columns)} new missingness indicator columns")
    else:
        print("\n   No Tier 2 features to process (all were dropped or no missing values in 30-90% range)")

    # =========================================================================
    # STEP 2.3: Calculate Imputation Values
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.3: CALCULATE IMPUTATION VALUES")
    print("=" * 80)

    print("\nCalculating imputation values for Tier 2 and Tier 3 features...")

    imputation_values = {}

    # Identify feature types among features we're keeping and have missing values
    features_with_missing = tier_2_to_process + [col for col in tier_3_impute if col in features_to_keep]

    # Categorical features - use mode (most common)
    categorical_features = [
        'formatted_work_type',
        'formatted_experience_level',
        'company_size',
        'state',
        'city',
        'remote_allowed'
    ]

    categorical_to_impute = [col for col in categorical_features if col in features_with_missing]

    if categorical_to_impute:
        print("\n📊 Categorical Features (impute with mode):")
        for col in categorical_to_impute:
            values = [row[col] for row in data if row.get(col) and row[col] != '']
            if values:
                mode_val = get_mode(values)
                imputation_values[col] = mode_val
                tier = "Tier 2" if col in tier_2_to_process else "Tier 3"
                print(f"   • {col:<30} → '{mode_val}' ({tier})")

    # Numerical features - use median
    numerical_features = [
        'views',
        'applies',
        'employee_count',
        'follower_count',
        'industry_count',
        'skill_count',
        'company_industry_count',
        'speciality_count',
        'benefit_count'
    ]

    numerical_to_impute = [col for col in numerical_features if col in features_with_missing]

    if numerical_to_impute:
        print("\n🔢 Numerical Features (impute with median):")
        for col in numerical_to_impute:
            values = []
            for row in data:
                if row.get(col) and row[col] != '':
                    try:
                        values.append(float(row[col]))
                    except:
                        pass
            if values:
                median_val = get_median(values)
                imputation_values[col] = median_val
                tier = "Tier 2" if col in tier_2_to_process else "Tier 3"
                print(f"   • {col:<30} → {median_val:.0f} ({tier})")

    # Text features - use 'Unknown' or ''
    text_features = [
        'company_name',
        'description_company',
        'primary_industry',
        'all_industries',
        'primary_skill',
        'all_skills',
        'primary_company_industry',
        'all_company_industries',
        'primary_speciality',
        'all_specialities',
        'country',
        'benefits_list'
    ]

    text_to_impute = [col for col in text_features if col in features_with_missing]

    if text_to_impute:
        print("\n📝 Text Features (impute with 'Unknown'):")
        for col in text_to_impute:
            imputation_values[col] = 'Unknown'
            tier = "Tier 2" if col in tier_2_to_process else "Tier 3"
            print(f"   • {col:<30} → 'Unknown' ({tier})")

    # =========================================================================
    # STEP 2.4: Apply Imputation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.4: APPLY IMPUTATION")
    print("=" * 80)

    print("\nImputing missing values for Tier 2 and Tier 3 features...")

    imputation_counts = {col: 0 for col in imputation_values}

    for row in data:
        for col, impute_val in imputation_values.items():
            if not row.get(col) or row[col] == '':
                row[col] = str(impute_val) if impute_val is not None else ''
                imputation_counts[col] += 1

    print(f"\nImputation summary (sorted by count):")
    for col, count in sorted(imputation_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            tier = "Tier 2" if col in tier_2_to_process else "Tier 3"
            print(f"   • {col:<30} {count:>6,} values imputed ({tier})")

    # =========================================================================
    # STEP 2.5: Save Clean Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2.5: SAVE CLEAN DATASET WITH MISSINGNESS INDICATORS")
    print("=" * 80)

    output_file = 'salary_data_no_missing.csv'
    print(f"\nSaving to {output_file}...")

    # Final fieldnames = kept features + new indicator columns
    final_fieldnames = features_to_keep + indicator_columns

    print(f"\nFinal feature count: {len(final_fieldnames)}")
    print(f"   Original features kept:        {len(features_to_keep)}")
    print(f"   New missingness indicators:    {len(indicator_columns)}")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    print(f"\n✓ Saved: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE! (IMPUTE + INDICATOR APPROACH)")
    print("=" * 80)

    # Check for remaining missing values
    remaining_missing = {}
    for col in final_fieldnames:
        missing_count = sum(1 for row in data if not row.get(col) or row[col] == '')
        if missing_count > 0:
            remaining_missing[col] = missing_count

    print(f"\n📊 SUMMARY:")
    print(f"   Input:  salary_data_final.csv ({len(data):,} records, {len(all_fields)} features)")
    print(f"   Output: {output_file} ({len(data):,} records, {len(final_fieldnames)} features)")
    print(f"")
    print(f"   Tier 1 (Dropped):               {len(features_to_drop)} features")
    print(f"   Tier 2 (Imputed+Indicator):     {len(tier_2_to_process)} features → +{len(indicator_columns)} indicators")
    print(f"   Tier 3 (Imputed only):          {len([c for c in tier_3_impute if c in features_to_keep])} features")
    print(f"   Total values imputed:           {sum(imputation_counts.values()):,}")

    if remaining_missing:
        print(f"\n⚠️  Features still with missing values:")
        for col, count in sorted(remaining_missing.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / len(data) * 100
            print(f"   • {col:<30} {count:>6,} ({pct:>5.1f}%)")
    else:
        print(f"\n✅ NO MISSING VALUES! Dataset is complete.")

    if indicator_columns:
        print(f"\n🎯 NEW MISSINGNESS INDICATOR FEATURES ({len(indicator_columns)}):")
        print("   (Binary columns that preserve the signal that a value was missing)")
        for ind_col in indicator_columns:
            original_col = ind_col.replace('_missing', '')
            count = indicator_counts.get(original_col, 0)
            pct = count / len(data) * 100
            print(f"   • {ind_col:<35} ({count:,} = 1, {pct:.1f}%)")

    print(f"\n📋 FEATURE BREAKDOWN ({len(final_fieldnames)} total):")

    feature_categories = {
        'Target': ['salary_normalized'],
        'Missingness Indicators': indicator_columns[:5],
        'Text': ['title', 'description', 'company_name', 'location', 'all_skills'],
        'Categorical': ['formatted_work_type', 'formatted_experience_level', 'state', 'city', 'company_size'],
        'Numerical': ['views', 'employee_count', 'follower_count', 'skill_count']
    }

    for category, examples in feature_categories.items():
        matching = [f for f in examples if f in final_fieldnames]
        if matching:
            if category == 'Missingness Indicators':
                print(f"\n   {category}: {len(indicator_columns)} features (showing first {len(matching)})")
            else:
                print(f"\n   {category}: {len(matching)} features")
            for f in matching[:5]:
                print(f"      • {f}")

    print("\n" + "=" * 80)
    print("WHY THIS APPROACH WORKS:")
    print("=" * 80)
    print("""
✓ Linear Regression: Can use imputed values + learn to weight them via indicators
✓ Random Forest/XGBoost: Can split on missingness indicators for better predictions
✓ Preserves Signal: Missingness itself can be predictive (MNAR patterns)
✓ Example: If 'applies_missing=1' correlates with higher salaries for senior roles,
           models can learn this pattern!
    """)

    print("\n" + "=" * 80)
    print("NEXT STEP: Step 3 - Encode Categorical Variables")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
