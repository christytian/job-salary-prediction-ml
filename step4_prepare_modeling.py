"""
Step 4: Prepare Data for Modeling
==================================
Prepare the final dataset with NLP features for machine learning models.

Tasks:
1. Load data with NLP features
2. Separate features (X) and target (y = salary_normalized)
3. Encode categorical features (One-Hot + Label Encoding)
4. Scale numerical features (StandardScaler)
5. Train/Test Split
6. Save preprocessed data and preprocessors

Input:  salary_data_with_nlp_features.csv (1,004 features)
Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv, preprocessors.pkl
"""

import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import Counter


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "test_size": 0.2,  # 20% for testing
    "random_state": 42,  # For reproducibility
    "stratify": False,  # Can't stratify on continuous target, but can try quantile-based
    # Features to drop (already converted to TF-IDF/Embeddings)
    "drop_text_features": True,  # Drop original text columns (already have TF-IDF/Embeddings)
    # Categorical features encoding strategy
    "onehot_threshold": 10,  # Use One-Hot if unique values < 10
    "label_threshold": 50,  # Use Label Encoding if unique values >= 10
    # Scaling
    "scale_numerical": True,  # Scale original numerical features
    "scale_tfidf": False,  # TF-IDF is already normalized
    "scale_embeddings": False,  # Embeddings are already normalized
}


def identify_feature_types(row_sample, fieldnames):
    """
    Identify feature types based on column names and sample data.

    Returns:
        dict with lists of feature names by type
    """
    categories = {
        "target": [],
        "categorical": [],
        "numerical_original": [],
        "missing_indicators": [],
        "tfidf": [],
        "embeddings": [],
        "text_original": [],
        "to_drop": [],
    }

    # Identify target
    categories["target"] = ["salary_normalized"]

    # Features to drop (redundant or not useful)
    redundant_features = [
        "normalized_salary",  # Duplicate of salary_normalized
        "salary_yearly_min",  # Redundant (used to create salary_normalized)
        "salary_yearly_max",  # Redundant
        "salary_yearly_med",  # Redundant
        "currency",  # Mostly USD, not predictive
    ]

    # Original text features (already converted to TF-IDF/Embeddings)
    text_features_original = [
        "title",
        "description",
        "all_skills",
        "company_name",
        "location",
        "description_company",
        "primary_industry",
        "all_industries",
        "primary_skill",
        "primary_company_industry",
        "all_company_industries",
        "primary_speciality",
        "all_specialities",
        "benefits_list",
        "country",
    ]

    # Categorical features (low cardinality)
    categorical_onehot = [
        "formatted_work_type",
        "formatted_experience_level",
        "company_size",
    ]

    # Categorical features (high cardinality - use Label Encoding)
    categorical_label = ["state", "city"]

    # Boolean features (already 0/1)
    boolean_features = ["remote_allowed"]

    # Original numerical features (need scaling)
    numerical_original = [
        "views",
        "applies",
        "employee_count",
        "follower_count",
        "industry_count",
        "skill_count",
        "benefit_count",
        "company_industry_count",
        "speciality_count",
    ]

    # Classify each feature
    for feat in fieldnames:
        if feat in categories["target"]:
            continue
        elif feat in redundant_features:
            categories["to_drop"].append(feat)
        elif feat in text_features_original and CONFIG["drop_text_features"]:
            categories["text_original"].append(feat)
            categories["to_drop"].append(feat)
        elif feat in categorical_onehot:
            categories["categorical"].append(feat)
        elif feat in categorical_label:
            categories["categorical"].append(feat)
        elif feat in boolean_features:
            categories["numerical_original"].append(
                feat
            )  # Already numeric, can include in X
        elif feat in numerical_original:
            categories["numerical_original"].append(feat)
        elif feat.endswith("_missing"):
            categories["missing_indicators"].append(feat)
        elif "tfidf" in feat.lower():
            categories["tfidf"].append(feat)
        elif "emb_" in feat.lower():
            categories["embeddings"].append(feat)
        else:
            # Default: keep it as numerical
            try:
                val = row_sample.get(feat, "")
                float(val)  # Check if can convert to float
                categories["numerical_original"].append(feat)
            except (ValueError, TypeError):
                # Can't convert, might be text or categorical
                categories["to_drop"].append(feat)

    return categories


def load_data(filename):
    """Load CSV data and return as list of dictionaries."""
    print(f"\nLoading {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = list(reader.fieldnames)

    print(f"✓ Loaded: {len(data):,} records, {len(fieldnames)} features")
    return data, fieldnames


def main():
    print("=" * 80)
    print("STEP 4: PREPARE DATA FOR MODELING")
    print("=" * 80)

    print(
        """
    This script prepares the final dataset for machine learning:
    
    1. Load data with NLP features (1,004 features)
    2. Separate features (X) and target (y = salary_normalized)
    3. Encode categorical features (One-Hot + Label Encoding)
    4. Scale numerical features (StandardScaler)
    5. Train/Test Split (80/20)
    6. Save preprocessed data and preprocessors
    
    Output:
    - X_train.csv, X_test.csv (features)
    - y_train.csv, y_test.csv (target)
    - preprocessors.pkl (scalers, encoders for future use)
    """
    )

    # =========================================================================
    # STEP 4.1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.1: LOAD DATA")
    print("=" * 80)

    data, fieldnames = load_data("salary_data_with_nlp_features.csv")

    if len(data) == 0:
        print("❌ ERROR: No data loaded!")
        return

    # Sample row for feature type identification
    row_sample = data[0]

    # =========================================================================
    # STEP 4.2: Identify Feature Types
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.2: IDENTIFY FEATURE TYPES")
    print("=" * 80)

    feature_types = identify_feature_types(row_sample, fieldnames)

    print(f"\n📊 Feature Classification:")
    print(f"   Target:                  {len(feature_types['target'])} feature")
    print(f"   Categorical:             {len(feature_types['categorical'])} features")
    print(
        f"   Numerical (original):    {len(feature_types['numerical_original'])} features"
    )
    print(
        f"   Missing indicators:      {len(feature_types['missing_indicators'])} features"
    )
    print(f"   TF-IDF features:         {len(feature_types['tfidf'])} features")
    print(f"   Embedding features:      {len(feature_types['embeddings'])} features")
    print(f"   Text (to drop):          {len(feature_types['text_original'])} features")
    print(f"   Other (to drop):         {len(feature_types['to_drop'])} features")

    print(f"\n📋 Features to drop ({len(feature_types['to_drop'])}):")
    for feat in feature_types["to_drop"][:10]:
        print(f"   • {feat}")
    if len(feature_types["to_drop"]) > 10:
        print(f"   ... and {len(feature_types['to_drop']) - 10} more")

    # =========================================================================
    # STEP 4.3: Extract Target Variable
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.3: EXTRACT TARGET VARIABLE")
    print("=" * 80)

    target_col = "salary_normalized"

    # Extract target
    y = []
    valid_indices = []

    for i, row in enumerate(data):
        try:
            salary = float(row.get(target_col, ""))
            if salary > 0:  # Valid salary
                y.append(salary)
                valid_indices.append(i)
        except (ValueError, TypeError):
            pass

    print(f"\n✓ Extracted target: {target_col}")
    print(f"✓ Valid records: {len(y):,} / {len(data):,}")
    print(f"✓ Salary range: ${min(y):,.0f} - ${max(y):,.0f}")
    print(f"✓ Mean salary: ${np.mean(y):,.0f}")
    print(f"✓ Median salary: ${np.median(y):,.0f}")

    # Filter data to only valid records
    data = [data[i] for i in valid_indices]
    y = np.array(y)

    # =========================================================================
    # STEP 4.4: Extract Features (X)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.4: EXTRACT FEATURES")
    print("=" * 80)

    # Features to keep
    features_to_keep = (
        feature_types["categorical"]
        + feature_types["numerical_original"]
        + feature_types["missing_indicators"]
        + feature_types["tfidf"]
        + feature_types["embeddings"]
    )

    print(f"\n✓ Keeping {len(features_to_keep)} features for modeling")

    # Initialize feature matrix (will convert to numpy array)
    X_dict = {feat: [] for feat in features_to_keep}

    for row in data:
        for feat in features_to_keep:
            val = row.get(feat, "")
            X_dict[feat].append(val)

    # Convert to list of lists (each inner list is a row)
    X_list = []
    for i in range(len(data)):
        row = [X_dict[feat][i] for feat in features_to_keep]
        X_list.append(row)

    print(f"✓ Extracted {len(X_list):,} rows × {len(features_to_keep)} features")

    # =========================================================================
    # STEP 4.5: Encode Categorical Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.5: ENCODE CATEGORICAL FEATURES")
    print("=" * 80)

    # Separate categorical by cardinality
    categorical_onehot = []  # Low cardinality (< 10 unique values)
    categorical_label = []  # High cardinality (>= 10 unique values)

    for feat in feature_types["categorical"]:
        unique_values = len(
            set([row.get(feat, "") for row in data if row.get(feat, "") != ""])
        )
        if unique_values < CONFIG["onehot_threshold"]:
            categorical_onehot.append(feat)
        else:
            categorical_label.append(feat)

    print(f"\n📊 Categorical Encoding Strategy:")
    print(f"   One-Hot Encoding (low cardinality): {len(categorical_onehot)} features")
    for feat in categorical_onehot:
        unique = len(
            set([row.get(feat, "") for row in data if row.get(feat, "") != ""])
        )
        print(f"      • {feat:<30} ({unique} unique values)")

    print(f"\n   Label Encoding (high cardinality):  {len(categorical_label)} features")
    for feat in categorical_label:
        unique = len(
            set([row.get(feat, "") for row in data if row.get(feat, "") != ""])
        )
        print(f"      • {feat:<30} ({unique} unique values)")

    # Find indices of categorical features in features_to_keep
    cat_onehot_indices = [
        features_to_keep.index(f) for f in categorical_onehot if f in features_to_keep
    ]
    cat_label_indices = [
        features_to_keep.index(f) for f in categorical_label if f in features_to_keep
    ]

    # Prepare data for encoding
    X_array = np.array(X_list, dtype=object)

    # Encode: One-Hot Encoding (low cardinality)
    onehot_encoder = None
    onehot_features = []

    if categorical_onehot:
        print(
            f"\n📊 Applying One-Hot Encoding to {len(categorical_onehot)} features..."
        )

        # Extract categorical data
        X_cat_onehot = X_array[:, cat_onehot_indices]

        # Fit One-Hot Encoder
        onehot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="first"
        )
        X_onehot_encoded = onehot_encoder.fit_transform(X_cat_onehot)

        # Get feature names
        onehot_feature_names = []
        for i, feat in enumerate(categorical_onehot):
            categories = onehot_encoder.categories_[i]
            for cat in categories[1:]:  # Skip first (dropped due to drop='first')
                onehot_feature_names.append(f"{feat}_{cat}")

        onehot_features = onehot_feature_names
        print(f"   ✓ Created {len(onehot_features)} One-Hot encoded features")
        print(f"      Example: {onehot_features[:3]}")

    # Encode: Label Encoding (high cardinality)
    label_encoders = {}
    label_features = []

    if categorical_label:
        print(f"\n📊 Applying Label Encoding to {len(categorical_label)} features...")

        X_array_labeled = X_array.copy()

        for i, feat in enumerate(categorical_label):
            idx = features_to_keep.index(feat)

            # Extract column
            col_data = X_array[:, idx].astype(str)

            # Fit Label Encoder
            le = LabelEncoder()
            col_encoded = le.fit_transform(col_data)

            # Replace in array
            X_array_labeled[:, idx] = col_encoded

            # Store encoder
            label_encoders[feat] = le

            label_features.append(feat)

        X_array = X_array_labeled
        print(f"   ✓ Encoded {len(label_features)} features with Label Encoding")

    # Remove original categorical columns and add one-hot encoded columns
    if cat_onehot_indices or cat_label_indices:
        # Build new feature list
        new_features = []
        indices_to_remove = set(cat_onehot_indices + cat_label_indices)

        # Add non-categorical features
        for i, feat in enumerate(features_to_keep):
            if i not in indices_to_remove:
                new_features.append(feat)

        # Add one-hot encoded features
        new_features.extend(onehot_features)

        # Rebuild X_array
        # Keep non-categorical columns
        keep_indices = [
            i for i in range(len(features_to_keep)) if i not in indices_to_remove
        ]
        X_non_cat = X_array[:, keep_indices]

        # Concatenate with one-hot encoded features
        if len(onehot_features) > 0:
            X_final = np.hstack([X_non_cat, X_onehot_encoded])
        else:
            X_final = X_non_cat

        features_to_keep = new_features
        print(f"\n✓ Final feature count: {len(features_to_keep)} features")

    # =========================================================================
    # STEP 4.6: Convert to Numeric and Scale Numerical Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.6: CONVERT TO NUMERIC AND SCALE")
    print("=" * 80)

    # Convert all to float (handle any remaining strings)
    print("\nConverting all features to numeric...")
    X_numeric = np.zeros((len(data), len(features_to_keep)))

    for i in range(len(features_to_keep)):
        feat = features_to_keep[i]
        for j in range(len(data)):
            try:
                X_numeric[j, i] = float(X_final[j, i])
            except (ValueError, TypeError):
                X_numeric[j, i] = 0.0  # Default to 0 if can't convert

    print(f"✓ Converted to numeric array: {X_numeric.shape}")

    # Identify which features need scaling
    # Numerical original features (need scaling)
    # TF-IDF and Embeddings (already normalized, don't scale)
    # Missing indicators (already 0/1, don't need scaling but can scale)

    numerical_to_scale = []
    scale_indices = []

    for i, feat in enumerate(features_to_keep):
        if feat in feature_types["numerical_original"]:
            numerical_to_scale.append(feat)
            scale_indices.append(i)
        elif feat in feature_types["missing_indicators"]:
            # Don't scale missing indicators (they're already 0/1)
            pass
        elif (
            feat.startswith(("title_tfidf_", "desc_tfidf_", "skills_tfidf_"))
            and CONFIG["scale_tfidf"]
        ):
            numerical_to_scale.append(feat)
            scale_indices.append(i)
        elif (
            feat.startswith(("title_emb_", "desc_emb_")) and CONFIG["scale_embeddings"]
        ):
            numerical_to_scale.append(feat)
            scale_indices.append(i)

    print("\n📊 Scaling Strategy:")
    print(f"   Features to scale: {len(numerical_to_scale)} features")
    if numerical_to_scale:
        print(f"      Examples: {numerical_to_scale[:5]}")
    print(
        f"   Features NOT scaled: {len(features_to_keep) - len(numerical_to_scale)} features"
    )
    print("      (TF-IDF, Embeddings, Missing indicators are already normalized)")

    # Apply StandardScaler
    scaler = None
    if CONFIG["scale_numerical"] and len(numerical_to_scale) > 0:
        print(
            f"\n📊 Applying StandardScaler to {len(numerical_to_scale)} numerical features..."
        )

        scaler = StandardScaler()

        # Scale only selected features
        X_scaled = X_numeric.copy()
        X_scaled[:, scale_indices] = scaler.fit_transform(X_numeric[:, scale_indices])

        print(f"   ✓ Scaled {len(numerical_to_scale)} features")
        print(
            f"      Mean after scaling: {np.mean(X_scaled[:, scale_indices]):.6f} (should be ~0)"
        )
        print(
            f"      Std after scaling:  {np.std(X_scaled[:, scale_indices]):.6f} (should be ~1)"
        )
    else:
        X_scaled = X_numeric
        print("\n⚠️  Scaling disabled in CONFIG or no features to scale")

    # =========================================================================
    # STEP 4.7: Train/Test Split
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.7: TRAIN/TEST SPLIT")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        shuffle=True,
    )

    print("\n✓ Train/Test Split:")
    print(
        f"   Training set: {len(X_train):,} samples ({100*(1-CONFIG['test_size']):.0f}%)"
    )
    print(f"   Test set:     {len(X_test):,} samples ({100*CONFIG['test_size']:.0f}%)")
    print(f"   Features:     {len(features_to_keep)} features")

    print("\n📊 Target Distribution:")
    print(
        f"   Train - Mean: ${np.mean(y_train):,.0f}, Median: ${np.median(y_train):,.0f}"
    )
    print(
        f"   Test  - Mean: ${np.mean(y_test):,.0f}, Median: ${np.median(y_test):,.0f}"
    )

    # =========================================================================
    # STEP 4.8: Save Data and Preprocessors
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4.8: SAVE DATA AND PREPROCESSORS")
    print("=" * 80)

    # Save X_train, X_test, y_train, y_test as CSV
    print("\nSaving preprocessed data...")

    # Save X_train
    with open("X_train.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features_to_keep)  # Header
        writer.writerows(X_train)
    print("   ✓ Saved: X_train.csv")

    # Save X_test
    with open("X_test.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features_to_keep)  # Header
        writer.writerows(X_test)
    print("   ✓ Saved: X_test.csv")

    # Save y_train
    with open("y_train.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([target_col])  # Header
        writer.writerows([[val] for val in y_train])
    print("   ✓ Saved: y_train.csv")

    # Save y_test
    with open("y_test.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([target_col])  # Header
        writer.writerows([[val] for val in y_test])
    print("   ✓ Saved: y_test.csv")

    # Save preprocessors
    preprocessors = {
        "scaler": scaler,
        "onehot_encoder": onehot_encoder,
        "label_encoders": label_encoders,
        "feature_names": features_to_keep,
        "categorical_onehot": categorical_onehot,
        "categorical_label": categorical_label,
        "numerical_to_scale": numerical_to_scale,
    }

    with open("preprocessors.pkl", "wb") as f:
        pickle.dump(preprocessors, f)
    print("   ✓ Saved: preprocessors.pkl")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4 COMPLETE!")
    print("=" * 80)

    print(
        f"""
📊 FINAL DATASET SUMMARY:

Input:
   • salary_data_with_nlp_features.csv ({len(data):,} records, 1,004 features)

Output:
   • X_train.csv ({len(X_train):,} samples × {len(features_to_keep)} features)
   • X_test.csv ({len(X_test):,} samples × {len(features_to_keep)} features)
   • y_train.csv ({len(y_train):,} samples)
   • y_test.csv ({len(y_test):,} samples)
   • preprocessors.pkl (scalers, encoders for future predictions)

Feature Breakdown:
   • Total features: {len(features_to_keep)}
   • Categorical (One-Hot): {len(onehot_features)}
   • Categorical (Label): {len(label_features)}
   • Numerical (scaled): {len(numerical_to_scale)}
   • TF-IDF: {len(feature_types['tfidf'])} (not scaled)
   • Embeddings: {len(feature_types['embeddings'])} (not scaled)
   • Missing indicators: {len(feature_types['missing_indicators'])} (not scaled)

Target:
   • Variable: {target_col}
   • Range: ${min(y):,.0f} - ${max(y):,.0f}
   • Mean: ${np.mean(y):,.0f}
   • Median: ${np.median(y):,.0f}

🚀 NEXT STEP:
   → Step 5: Train baseline models (Linear Regression, Ridge, Lasso)
   → Step 6: Train advanced models (Random Forest, XGBoost)
   → Step 7: Evaluate and compare models

Ready for modeling! 🎯
    """
    )

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
