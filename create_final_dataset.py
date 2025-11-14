"""
Complete Preprocessing Pipeline
================================
From salary_data_cleaned.csv to final modeling-ready dataset

Steps:
1. Load cleaned data
2. Create normalized salary column
3. Drop records with invalid salaries
4. Drop non-essential features
5. Save final dataset
"""

import csv


def calculate_normalized_salary(row):
    """
    Calculate normalized salary from available salary data.
    Returns None if salary is invalid or missing.
    """
    min_sal = row.get('salary_yearly_min', '')
    max_sal = row.get('salary_yearly_max', '')
    med_sal = row.get('salary_yearly_med', '')

    # Convert to float
    try:
        min_val = float(min_sal) if min_sal and min_sal != '' else None
    except:
        min_val = None

    try:
        max_val = float(max_sal) if max_sal and max_sal != '' else None
    except:
        max_val = None

    try:
        med_val = float(med_sal) if med_sal and med_sal != '' else None
        # Filter out unrealistic median values (< $1,000/year)
        if med_val and med_val < 1000:
            med_val = None
    except:
        med_val = None

    # Calculate normalized salary
    if min_val and max_val:
        return (min_val + max_val) / 2
    elif med_val:
        return med_val
    elif min_val:
        return min_val * 1.15
    elif max_val:
        return max_val * 0.85
    else:
        return None


def main():
    print("\n" + "=" * 80)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    print("\nLoading salary_data_cleaned.csv...")
    with open('salary_data_cleaned.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        original_fields = list(reader.fieldnames)

    print(f"✓ Loaded: {len(data):,} records")
    print(f"✓ Original features: {len(original_fields)}")

    # =========================================================================
    # STEP 2: Create Normalized Salary
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: CREATE NORMALIZED SALARY")
    print("=" * 80)

    print("\nCalculating salary_normalized for each record...")
    print("  Formula: (min + max) / 2, or use median if available")

    normalized_count = 0
    invalid_count = 0

    stats = {
        'from_min_max': 0,
        'from_median': 0,
        'from_min_only': 0,
        'from_max_only': 0,
        'invalid': 0
    }

    for row in data:
        min_sal = row.get('salary_yearly_min', '')
        max_sal = row.get('salary_yearly_max', '')
        med_sal = row.get('salary_yearly_med', '')

        # Check what values exist
        has_min = min_sal and min_sal != ''
        has_max = max_sal and max_sal != ''
        has_med = med_sal and med_sal != ''
        try:
            has_valid_med = has_med and float(med_sal) >= 1000
        except:
            has_valid_med = False

        # Calculate normalized salary
        normalized = calculate_normalized_salary(row)

        if normalized:
            row['salary_normalized'] = f"{normalized:.2f}"
            normalized_count += 1

            # Track source
            if has_min and has_max:
                stats['from_min_max'] += 1
            elif has_valid_med:
                stats['from_median'] += 1
            elif has_min:
                stats['from_min_only'] += 1
            elif has_max:
                stats['from_max_only'] += 1
        else:
            row['salary_normalized'] = ''
            invalid_count += 1
            stats['invalid'] += 1

    print(f"\n✓ Successfully normalized: {normalized_count:,} records")
    print(f"✓ Invalid/missing: {invalid_count:,} records")

    print(f"\nNormalization breakdown:")
    print(f"  • From min & max:  {stats['from_min_max']:>6,} ({stats['from_min_max']/len(data)*100:.1f}%)")
    print(f"  • From median:     {stats['from_median']:>6,} ({stats['from_median']/len(data)*100:.1f}%)")
    print(f"  • From min only:   {stats['from_min_only']:>6,} ({stats['from_min_only']/len(data)*100:.1f}%)")
    print(f"  • From max only:   {stats['from_max_only']:>6,} ({stats['from_max_only']/len(data)*100:.1f}%)")
    print(f"  • Invalid:         {stats['invalid']:>6,} ({stats['invalid']/len(data)*100:.1f}%)")

    # =========================================================================
    # STEP 3: Drop Invalid Records
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: REMOVE INVALID RECORDS")
    print("=" * 80)

    print(f"\nRemoving {invalid_count:,} records with invalid/missing salary data...")

    # Filter to keep only valid records
    valid_data = [row for row in data if row.get('salary_normalized') and row['salary_normalized'] != '']

    print(f"✓ Kept: {len(valid_data):,} records with valid salary_normalized")
    print(f"✓ Removed: {len(data) - len(valid_data):,} records")

    # =========================================================================
    # STEP 4: Drop Non-Essential Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DROP NON-ESSENTIAL FEATURES")
    print("=" * 80)

    # Define features to drop
    features_to_drop = [
        # IDs (not predictive)
        'job_id',
        'salary_id',
        'company_id',

        # URLs (not useful for modeling)
        'job_posting_url',
        'application_url',
        'url',

        # Timestamps (temporal features not needed)
        'original_listed_time',
        'listed_time',
        'expiry',
        'closed_time',
        'time_recorded',

        # Redundant salary columns (now have salary_normalized)
        'pay_period',
        'pay_period_salary',
        'compensation_type',
        'compensation_type_salary',
        'currency_salary',
        'max_salary_salary',
        'med_salary_salary',
        'min_salary_salary',

        # Redundant work type (keep formatted_work_type)
        'work_type',

        # Other non-essential
        'posting_domain',
        'application_type',
        'sponsored',
        'fips',
        'zip_code',
        'zip_code_company',
        'address',
    ]

    print(f"\nDropping {len(features_to_drop)} non-essential features:")
    print("\nCategories:")
    print(f"  • IDs: job_id, salary_id, company_id")
    print(f"  • URLs: job_posting_url, application_url, url")
    print(f"  • Timestamps: 5 features")
    print(f"  • Redundant salary columns: 8 features")
    print(f"  • Other non-essential: 8 features")

    # Create list of features to keep
    features_to_keep = [f for f in original_fields if f not in features_to_drop]

    # Add salary_normalized if not already there
    if 'salary_normalized' not in features_to_keep:
        # Insert after salary_yearly_med
        if 'salary_yearly_med' in features_to_keep:
            idx = features_to_keep.index('salary_yearly_med') + 1
            features_to_keep.insert(idx, 'salary_normalized')
        else:
            features_to_keep.append('salary_normalized')

    print(f"\n✓ Kept: {len(features_to_keep)} essential features")
    print(f"✓ Dropped: {len(features_to_drop)} non-essential features")

    # =========================================================================
    # STEP 5: Save Final Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SAVE FINAL DATASET")
    print("=" * 80)

    output_file = 'salary_data_final.csv'
    print(f"\nSaving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=features_to_keep, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(valid_data)

    print(f"✓ Saved: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    # Calculate final statistics
    all_salaries = []
    for row in valid_data:
        try:
            all_salaries.append(float(row['salary_normalized']))
        except:
            pass

    all_salaries.sort()

    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"   Input file:  salary_data_cleaned.csv ({len(data):,} records)")
    print(f"   Output file: {output_file} ({len(valid_data):,} records)")
    print(f"   Features:    {len(original_fields)} → {len(features_to_keep)}")
    print(f"   Records removed: {len(data) - len(valid_data):,} (invalid salaries)")

    print(f"\n💰 SALARY STATISTICS (salary_normalized):")
    if all_salaries:
        print(f"   Count:    {len(all_salaries):,}")
        print(f"   Min:      ${min(all_salaries):,.0f}")
        print(f"   25th:     ${all_salaries[len(all_salaries)//4]:,.0f}")
        print(f"   Median:   ${all_salaries[len(all_salaries)//2]:,.0f}")
        print(f"   75th:     ${all_salaries[3*len(all_salaries)//4]:,.0f}")
        print(f"   Max:      ${max(all_salaries):,.0f}")
        print(f"   Mean:     ${sum(all_salaries)/len(all_salaries):,.0f}")

    print(f"\n📋 KEY FEATURES KEPT ({len(features_to_keep)} total):")

    feature_categories = {
        'Target': ['salary_normalized', 'salary_yearly_min', 'salary_yearly_max', 'salary_yearly_med'],
        'Text': ['title', 'description', 'skills_desc', 'all_skills'],
        'Categorical': ['formatted_work_type', 'formatted_experience_level', 'state', 'city',
                       'company_size', 'remote_allowed'],
        'Numerical': ['views', 'applies', 'employee_count', 'follower_count'],
    }

    for category, features in feature_categories.items():
        matching = [f for f in features if f in features_to_keep]
        if matching:
            print(f"\n   {category}:")
            for f in matching:
                print(f"     • {f}")

    print("\n" + "=" * 80)
    print("✅ READY FOR MODELING!")
    print("=" * 80)
    print(f"""
Next steps:
1. Feature engineering (text processing, encoding)
2. Handle missing values in remaining features
3. Scale numerical features
4. Train/test split
5. Build baseline model (Linear Regression)
6. Build advanced models (Random Forest, XGBoost)
7. Evaluate and compare

Your target variable: salary_normalized
Dataset: {output_file}
Records: {len(valid_data):,}
Features: {len(features_to_keep)}
    """)


if __name__ == "__main__":
    main()
