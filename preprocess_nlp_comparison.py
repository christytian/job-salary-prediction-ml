"""
Preprocess NLP Comparison Data
===============================
Preprocess three NLP feature files (TF-IDF, SBERT, Hybrid) for model comparison.

For each NLP method:
1. Load data with NLP features
2. Separate features (X) and target (y = salary_normalized)
3. Encode categorical features (One-Hot + Label Encoding)
4. Scale numerical features (StandardScaler)
5. Train/Test Split (80/20)
6. Save preprocessed data and preprocessors

Input files:
- nlp_features/salary_data_nlp_tfidf.csv
- nlp_features/salary_data_nlp_sbert.csv
- nlp_features/salary_data_nlp_hybrid.csv

Output (for each method):
- nlp_features/X_train_{method}.csv, X_test_{method}.csv
- nlp_features/y_train_{method}.csv, y_test_{method}.csv
- nlp_features/preprocessors_{method}.pkl
"""

import csv
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "test_size": 0.2,  # 20% for testing (80/20 split)
    "random_state": 42,
    "input_dir": "nlp_features",
    "output_dir": "preprocessed_data",  # New folder for preprocessed data
    # Features to drop (already converted to NLP features)
    "drop_text_features": True,
    # Categorical features encoding strategy
    "categorical_onehot": [
        "formatted_work_type",
        "formatted_experience_level",
        "company_size",
    ],
    "categorical_label": ["state", "city"],
    # Numerical features to scale
    "numerical_to_scale": [
        "views",
        "applies",
        "employee_count",
        "follower_count",
        "industry_count",
        "skill_count",
        "benefit_count",
        "company_industry_count",
        "speciality_count",
    ],
    # Features to drop (redundant)
    "redundant_features": [
        "normalized_salary",  # Duplicate of salary_normalized
        "currency",  # Mostly USD, not predictive
    ],
    # Original text features (already converted to NLP)
    "text_features_to_drop": [
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
    ],
}

# NLP methods to process
NLP_METHODS = ["tfidf", "sbert", "hybrid"]


def load_data(filename):
    """Load CSV data and return as list of dictionaries."""
    print(f"\n📖 Loading {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
    
    print(f"   ✓ Loaded {len(data):,} records")
    print(f"   ✓ {len(fieldnames)} columns")
    
    return data, fieldnames


def identify_feature_types(fieldnames):
    """Identify feature types based on column names."""
    categories = {
        "target": [],
        "categorical_onehot": [],
        "categorical_label": [],
        "numerical_to_scale": [],
        "numerical_keep": [],
        "nlp_features": [],
        "to_drop": [],
    }
    
    # Target
    if "salary_normalized" in fieldnames:
        categories["target"] = ["salary_normalized"]
    
    # Categorical features
    categories["categorical_onehot"] = [
        f for f in CONFIG["categorical_onehot"] if f in fieldnames
    ]
    categories["categorical_label"] = [
        f for f in CONFIG["categorical_label"] if f in fieldnames
    ]
    
    # Numerical features
    categories["numerical_to_scale"] = [
        f for f in CONFIG["numerical_to_scale"] if f in fieldnames
    ]
    
    # NLP features
    for feat in fieldnames:
        if feat in categories["target"]:
            continue
        elif feat in CONFIG["redundant_features"]:
            categories["to_drop"].append(feat)
        elif feat in CONFIG["text_features_to_drop"] and CONFIG["drop_text_features"]:
            categories["to_drop"].append(feat)
        elif "tfidf" in feat.lower() or "emb_" in feat.lower():
            categories["nlp_features"].append(feat)
        elif feat.endswith("_missing"):
            categories["numerical_keep"].append(feat)  # Missing indicators
        elif feat in ["remote_allowed"]:
            categories["numerical_keep"].append(feat)  # Boolean
        elif feat not in categories["categorical_onehot"] and feat not in categories["categorical_label"]:
            # Check if it's numeric
            try:
                # Assume it's numeric if not in other categories
                if feat not in categories["numerical_to_scale"]:
                    categories["numerical_keep"].append(feat)
            except:
                pass
    
    return categories


def preprocess_method(input_file, method_name):
    """Preprocess data for a specific NLP method."""
    print("\n" + "=" * 80)
    print(f"PREPROCESSING: {method_name.upper()}")
    print("=" * 80)
    
    # Load data
    data, fieldnames = load_data(input_file)
    
    if len(data) == 0:
        print(f"❌ ERROR: No data loaded from {input_file}!")
        return None
    
    # Identify feature types
    feature_types = identify_feature_types(fieldnames)
    
    print(f"\n📊 Feature Classification:")
    print(f"   Target:              {len(feature_types['target'])}")
    print(f"   Categorical (OneHot): {len(feature_types['categorical_onehot'])}")
    print(f"   Categorical (Label):  {len(feature_types['categorical_label'])}")
    print(f"   Numerical (scale):    {len(feature_types['numerical_to_scale'])}")
    print(f"   Numerical (keep):     {len(feature_types['numerical_keep'])}")
    print(f"   NLP features:         {len(feature_types['nlp_features'])}")
    print(f"   To drop:              {len(feature_types['to_drop'])}")
    
    # =========================================================================
    # Separate X and y
    # =========================================================================
    print(f"\n📊 Separating features and target...")
    
    target_col = feature_types["target"][0]
    
    # Get all feature columns (excluding target and to_drop)
    feature_cols = [
        f
        for f in fieldnames
        if f not in feature_types["target"] and f not in feature_types["to_drop"]
    ]
    
    print(f"   Features to keep: {len(feature_cols)}")
    
    # Extract X and y
    X_data = []
    y_data = []
    
    for row in data:
        try:
            # Extract features
            x_row = [row.get(f, '') for f in feature_cols]
            X_data.append(x_row)
            
            # Extract target
            y_val = row.get(target_col, '')
            if y_val:
                y_data.append(float(y_val))
            else:
                y_data.append(np.nan)
        except Exception as e:
            print(f"   ⚠️  Error processing row: {e}")
            continue
    
    X_array = np.array(X_data, dtype=object)
    y_array = np.array(y_data)
    
    # Remove rows with missing target
    valid_mask = ~np.isnan(y_array)
    X_array = X_array[valid_mask]
    y_array = y_array[valid_mask]
    
    print(f"   ✓ Valid samples: {len(y_array):,}")
    print(f"   ✓ Removed {len(data) - len(y_array):,} rows with missing target")
    
    # =========================================================================
    # Convert to numeric where possible
    # =========================================================================
    print(f"\n📊 Converting to numeric...")
    
    X_numeric = []
    for i, col_name in enumerate(feature_cols):
        col_data = X_array[:, i]
        
        # Try to convert to float
        numeric_col = []
        for val in col_data:
            try:
                numeric_col.append(float(val))
            except (ValueError, TypeError):
                # Keep as string for categorical
                numeric_col.append(str(val))
        
        X_numeric.append(numeric_col)
    
    X_array = np.array(X_numeric, dtype=object).T
    
    # =========================================================================
    # Encode categorical features
    # =========================================================================
    print(f"\n📊 Encoding categorical features...")
    
    # Find indices of categorical features
    cat_onehot_indices = [
        feature_cols.index(f) for f in feature_types["categorical_onehot"] if f in feature_cols
    ]
    cat_label_indices = [
        feature_cols.index(f) for f in feature_types["categorical_label"] if f in feature_cols
    ]
    
    # Label Encoding (in-place replacement)
    label_encoders = {}
    if cat_label_indices:
        print(f"   Applying Label Encoding to {len(cat_label_indices)} features...")
        
        for idx in cat_label_indices:
            feat_name = feature_cols[idx]
            col_data = X_array[:, idx].astype(str)
            
            # Fit Label Encoder
            le = LabelEncoder()
            col_encoded = le.fit_transform(col_data)
            
            # Replace in array
            X_array[:, idx] = col_encoded.astype(float)
            
            # Store encoder
            label_encoders[feat_name] = le
        
        print(f"   ✓ Label encoded {len(label_encoders)} features")
    
    # One-Hot Encoding
    onehot_encoder = None
    onehot_features = []
    if cat_onehot_indices:
        print(f"   Applying One-Hot Encoding to {len(cat_onehot_indices)} features...")
        
        # Extract categorical columns
        cat_onehot_data = []
        for idx in cat_onehot_indices:
            col_data = X_array[:, idx].astype(str)
            cat_onehot_data.append(col_data)
        
        # Combine for one-hot encoding
        cat_onehot_combined = np.array(cat_onehot_data).T
        
        # Fit One-Hot Encoder
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_onehot = onehot_encoder.fit_transform(cat_onehot_combined)
        
        # Get feature names
        onehot_feature_names = []
        for i, idx in enumerate(cat_onehot_indices):
            feat_name = feature_cols[idx]
            categories = onehot_encoder.categories_[i]
            for cat in categories:
                onehot_feature_names.append(f"{feat_name}_{cat}")
        
        onehot_features = onehot_feature_names
        print(f"   ✓ One-Hot encoded: {len(onehot_features)} features")
    
    # Remove categorical columns and add one-hot encoded
    if cat_onehot_indices or cat_label_indices:
        indices_to_remove = set(cat_onehot_indices + cat_label_indices)
        
        # Keep non-categorical columns
        keep_indices = [i for i in range(len(feature_cols)) if i not in indices_to_remove]
        X_non_cat = X_array[:, keep_indices]
        
        # Convert to float
        X_non_cat = X_non_cat.astype(float)
        
        # Add one-hot encoded
        if len(onehot_features) > 0:
            X_final = np.hstack([X_non_cat, X_onehot])
        else:
            X_final = X_non_cat
        
        # Update feature names
        new_feature_names = [feature_cols[i] for i in keep_indices] + onehot_features
    else:
        X_final = X_array.astype(float)
        new_feature_names = feature_cols
    
    print(f"   ✓ Final feature count: {X_final.shape[1]} features")
    
    # =========================================================================
    # Scale numerical features
    # =========================================================================
    print(f"\n📊 Scaling numerical features...")
    
    scaler = None
    if CONFIG["numerical_to_scale"]:
        # Find indices of numerical features to scale
        scale_indices = []
        for feat_name in CONFIG["numerical_to_scale"]:
            if feat_name in new_feature_names:
                scale_indices.append(new_feature_names.index(feat_name))
        
        if scale_indices:
            print(f"   Scaling {len(scale_indices)} numerical features...")
            
            # Extract columns to scale
            X_to_scale = X_final[:, scale_indices]
            
            # Fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_to_scale)
            
            # Replace scaled columns
            X_final[:, scale_indices] = X_scaled
            
            print(f"   ✓ Scaled {len(scale_indices)} features")
    
    # =========================================================================
    # Train/Test Split (80/20)
    # =========================================================================
    print(f"\n📊 Train/Test Split (80/20)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_array,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set:     {len(X_test):,} samples")
    print(f"   Features:     {X_train.shape[1]} features")
    
    # =========================================================================
    # Save preprocessed data
    # =========================================================================
    print(f"\n💾 Saving preprocessed data...")
    
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save X_train, X_test
    with open(os.path.join(output_dir, f"X_train_{method_name}.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_feature_names)
        writer.writerows(X_train)
    
    with open(os.path.join(output_dir, f"X_test_{method_name}.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_feature_names)
        writer.writerows(X_test)
    
    # Save y_train, y_test
    with open(os.path.join(output_dir, f"y_train_{method_name}.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([target_col])
        writer.writerows([[y] for y in y_train])
    
    with open(os.path.join(output_dir, f"y_test_{method_name}.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([target_col])
        writer.writerows([[y] for y in y_test])
    
    print(f"   ✓ Saved: X_train_{method_name}.csv, X_test_{method_name}.csv")
    print(f"   ✓ Saved: y_train_{method_name}.csv, y_test_{method_name}.csv")
    
    # =========================================================================
    # Save preprocessors
    # =========================================================================
    preprocessors = {
        "feature_names": new_feature_names,
        "scaler": scaler,
        "onehot_encoder": onehot_encoder,
        "label_encoders": label_encoders,
        "categorical_onehot": feature_types["categorical_onehot"],
        "categorical_label": feature_types["categorical_label"],
        "numerical_to_scale": CONFIG["numerical_to_scale"],
    }
    
    preprocessor_file = os.path.join(output_dir, f"preprocessors_{method_name}.pkl")
    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessors, f)
    
    print(f"   ✓ Saved: preprocessors_{method_name}.pkl")
    
    return {
        "method": method_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "target_mean": np.mean(y_train),
        "target_std": np.std(y_train),
    }


def main():
    print("=" * 80)
    print("PREPROCESS NLP COMPARISON DATA")
    print("=" * 80)
    print("Processing three NLP methods: TF-IDF, SBERT, Hybrid")
    print("Train/Test Split: 80/20")
    
    results = []
    
    for method in NLP_METHODS:
        input_file = os.path.join(CONFIG["input_dir"], f"salary_data_nlp_{method}.csv")
        
        if not os.path.exists(input_file):
            print(f"\n⚠️  Skipping {method}: {input_file} not found")
            continue
        
        result = preprocess_method(input_file, method)
        if result:
            results.append(result)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'Train':<10} {'Test':<10} {'Features':<12} {'Target Mean':<15} {'Target Std':<12}")
    print("-" * 80)
    
    for r in results:
        print(
            f"{r['method']:<15} {r['n_train']:<10,} {r['n_test']:<10,} {r['n_features']:<12} "
            f"${r['target_mean']:<14,.2f} ${r['target_std']:<11,.2f}"
        )
    
    print("\n✅ Preprocessing complete for all methods!")
    print(f"\n📁 Output files saved in: {CONFIG['output_dir']}/")


if __name__ == "__main__":
    main()

