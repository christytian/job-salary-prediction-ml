"""
Feature Engineering Step 1: Analyze Features and Missing Values
================================================================
Understand what data we have before engineering features
"""

import csv
from collections import Counter


def main():
    print("=" * 80)
    print("STEP 1: FEATURE ANALYSIS")
    print("=" * 80)

    # Load data
    with open('salary_data_final.csv', 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
        features = list(data[0].keys())

    print(f"\nDataset: {len(data):,} records, {len(features)} features")

    # Categorize features
    text_features = []
    categorical_features = []
    numerical_features = []
    target_features = []

    for f in features:
        if f in ['salary_normalized', 'salary_yearly_min', 'salary_yearly_max', 'salary_yearly_med']:
            target_features.append(f)
        elif f in ['title', 'description', 'skills_desc', 'all_skills', 'benefits_list',
                   'all_industries', 'all_specialities', 'company_name', 'location',
                   'description_company', 'primary_skill', 'primary_industry',
                   'primary_company_industry', 'primary_speciality']:
            text_features.append(f)
        elif f in ['formatted_work_type', 'formatted_experience_level', 'state',
                   'city', 'company_size', 'remote_allowed']:
            categorical_features.append(f)
        elif f in ['views', 'applies', 'employee_count', 'follower_count',
                   'skill_count', 'benefit_count', 'industry_count',
                   'company_industry_count', 'speciality_count']:
            numerical_features.append(f)

    print("\n" + "=" * 80)
    print("MISSING VALUE ANALYSIS")
    print("=" * 80)

    missing_info = []
    for col in features:
        missing_count = sum(1 for row in data if not row.get(col) or row[col] == '')
        if missing_count > 0:
            pct = missing_count / len(data) * 100
            missing_info.append((col, missing_count, pct))

    missing_info.sort(key=lambda x: x[2], reverse=True)

    # Tier-based strategy for missing values
    tier_1_drop = []      # >90% missing
    tier_2_indicator = []  # 30-90% missing (impute + add indicator)
    tier_3_impute = []     # <30% missing (impute only)

    print(f"\n{'Feature':<35} {'Missing':>10} {'Percent':>10} {'Strategy'}")
    print("-" * 80)

    for col, count, pct in missing_info:
        if pct >= 90:
            action = " TIER 1: DROP"
            tier_1_drop.append((col, pct))
        elif pct >= 30:
            action = "  TIER 2: IMPUTE + INDICATOR"
            tier_2_indicator.append((col, pct))
        else:
            action = " TIER 3: IMPUTE ONLY"
            tier_3_impute.append((col, pct))

        print(f"{col:<35} {count:>10,} {pct:>9.1f}% {action}")

    print("\n" + "=" * 80)
    print("FEATURE TYPE BREAKDOWN")
    print("=" * 80)

    print(f"\n TARGET: 1 feature")
    print(f"   • salary_normalized")

    print(f"\n TEXT FEATURES: {len(text_features)} features")
    for f in text_features:
        non_empty = sum(1 for row in data if row.get(f) and row[f] != '')
        pct = non_empty / len(data) * 100
        print(f"   • {f:<30} ({pct:>5.1f}% complete)")

    print(f"\n  CATEGORICAL FEATURES: {len(categorical_features)} features")
    for f in categorical_features:
        non_empty = sum(1 for row in data if row.get(f) and row[f] != '')
        unique = len(set(row[f] for row in data if row.get(f) and row[f] != ''))
        pct = non_empty / len(data) * 100
        print(f"   • {f:<30} {unique:>4} unique ({pct:>5.1f}% complete)")

    print(f"\n NUMERICAL FEATURES: {len(numerical_features)} features")
    for f in numerical_features:
        non_empty = sum(1 for row in data if row.get(f) and row[f] != '')
        pct = non_empty / len(data) * 100
        print(f"   • {f:<30} ({pct:>5.1f}% complete)")

    print("\n" + "=" * 80)
    print("TIER-BASED MISSING VALUE STRATEGY")
    print("=" * 80)

    print("""
Strategy: Preserve missingness signal for predictive power!

WHY? Missing values can be informative (MNAR - Missing Not At Random)
- Missing 'applies' might indicate newer jobs  different salary patterns
- Missing 'remote_allowed' might indicate older postings or certain industries
- Missingness indicators let models learn these patterns!
    """)

    if tier_1_drop:
        print("\n TIER 1: DROP (≥90% missing) - Too sparse to be useful")
        print("-" * 80)
        for col, pct in tier_1_drop:
            print(f"   • {col:<30} ({pct:.1f}% missing)")

    if tier_2_indicator:
        print("\n  TIER 2: IMPUTE + ADD INDICATOR (30-90% missing)")
        print("-" * 80)
        print("    Impute with median/mode AND add '{feature}_missing' binary column")
        print("    Preserves the signal that value was missing")
        for col, pct in tier_2_indicator:
            print(f"   • {col:<30} ({pct:.1f}% missing)  Add '{col}_missing'")

    if tier_3_impute:
        print("\n TIER 3: IMPUTE ONLY (<30% missing)")
        print("-" * 80)
        print("    Simple imputation with median/mode (no indicator needed)")
        for col, pct in tier_3_impute:
            print(f"   • {col:<30} ({pct:.1f}% missing)")

    print(f"\n SUMMARY:")
    print(f"   Tier 1 (Drop):              {len(tier_1_drop)} features")
    print(f"   Tier 2 (Impute+Indicator):  {len(tier_2_indicator)} features  +{len(tier_2_indicator)} new indicator columns")
    print(f"   Tier 3 (Impute only):       {len(tier_3_impute)} features")

    print("\n" + "=" * 80)
    print("Next: Run step2_handle_missing_values.py to implement this strategy")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
