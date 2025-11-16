"""
Optimized Categorical Feature Encoding
======================================

This script performs optimized encoding on categorical features in the hybrid NLP features file.
It uses different encoding strategies based on feature cardinality and distribution.

Input:  salary_data_with_nlp_hybrid_features.csv
Output: salary_data_with_onehot_features.csv

Encoding Strategies:
- Low cardinality (<10 categories): One-Hot Encoding
- High cardinality (city, state): Frequency Encoding
- Binary simplification (country): US vs Non-US
- Drop (currency): Removed (99.97% USD)

Process:
1. Load the hybrid features CSV
2. Identify categorical features
3. Apply optimized encoding strategies
4. Keep all numeric features (including NLP features)
5. Save to new CSV file
"""

import csv
import sys
import os
import math
from typing import Dict, List
from collections import Counter

try:
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Will use manual one-hot encoding.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features to encode with One-Hot (low cardinality)
ONEHOT_FEATURES = [
    'formatted_work_type',
    'formatted_experience_level',
    'company_size'
]

# Features to encode with Frequency Encoding (high cardinality)
FREQUENCY_FEATURES = [
    'city',
    'state'
]

# Features to encode as Binary
BINARY_FEATURES = [
    'country'  # US vs Non-US
]

# Features to drop (almost no information)
DROP_FEATURES = [
    'currency'  # 99.97% USD
]

# All categorical features
CATEGORICAL_FEATURES = ONEHOT_FEATURES + FREQUENCY_FEATURES + BINARY_FEATURES + DROP_FEATURES

# State name normalization mapping (full name -> abbreviation)
STATE_NORMALIZATION = {
    'California': 'CA',
    'New York': 'NY',
    'Texas': 'TX',
    'Illinois': 'IL',
    'Florida': 'FL',
    'Pennsylvania': 'PA',
    'Ohio': 'OH',
    'Georgia': 'GA',
    'North Carolina': 'NC',
    'Michigan': 'MI',
    'New Jersey': 'NJ',
    'Virginia': 'VA',
    'Washington': 'WA',
    'Arizona': 'AZ',
    'Massachusetts': 'MA',
    'Tennessee': 'TN',
    'Indiana': 'IN',
    'Missouri': 'MO',
    'Maryland': 'MD',
    'Wisconsin': 'WI',
    'Colorado': 'CO',
    'Minnesota': 'MN',
    'South Carolina': 'SC',
    'Alabama': 'AL',
    'Louisiana': 'LA',
    'Kentucky': 'KY',
    'Oregon': 'OR',
    'Oklahoma': 'OK',
    'Connecticut': 'CT',
    'Utah': 'UT',
    'Iowa': 'IA',
    'Nevada': 'NV',
    'Arkansas': 'AR',
    'Mississippi': 'MS',
    'Kansas': 'KS',
    'New Mexico': 'NM',
    'Nebraska': 'NE',
    'West Virginia': 'WV',
    'Idaho': 'ID',
    'Hawaii': 'HI',
    'New Hampshire': 'NH',
    'Maine': 'ME',
    'Montana': 'MT',
    'Rhode Island': 'RI',
    'Delaware': 'DE',
    'South Dakota': 'SD',
    'North Dakota': 'ND',
    'Alaska': 'AK',
    'Vermont': 'VT',
    'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# Text fields that were converted to NLP features (can be dropped after encoding)
TEXT_FIELDS_TO_DROP = [
    'company_name',
    'title',
    'description',
    'location',
    'description_company',
    'primary_industry',
    'all_industries',
    'primary_skill',
    'all_skills',
    'benefits_list',
    'primary_company_industry',
    'all_company_industries',
    'primary_speciality',
    'all_specialities'
]

# Features to always keep (target variable and important indicators)
FEATURES_TO_KEEP = [
    'salary_normalized',
    'benefits_list_missing'
]


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def normalize_state(state: str) -> str:
    """
    Normalize state names to abbreviations.
    Handles duplicates like "California" -> "CA", "New York" -> "NY"
    
    Args:
        state: State name or abbreviation
    
    Returns:
        Normalized state abbreviation
    """
    if not state or state.strip() == '' or state == '0':
        return 'missing'
    
    state = state.strip()
    
    # If already an abbreviation (2 letters), return as is
    if len(state) == 2 and state.isalpha():
        return state.upper()
    
    # Check if it's in the normalization mapping
    if state in STATE_NORMALIZATION:
        return STATE_NORMALIZATION[state]
    
    # Return original if not found
    return state


def frequency_encode(values: List[str]) -> List[float]:
    """
    Frequency encoding: replace categories with their frequency (count / total).
    Uses log(count + 1) to handle rare categories better.
    
    Args:
        values: List of categorical values
        feature_name: Name of the feature
    
    Returns:
        List of frequency-encoded values
    """
    # Count frequencies
    counter = Counter(values)
    total = len(values)
    
    # Create frequency mapping (using log to reduce impact of rare categories)
    frequency_map = {}
    for category, count in counter.items():
        # Use log(count + 1) to handle rare categories
        frequency_map[category] = math.log(count + 1) / math.log(total + 1)
    
    # Encode values
    encoded = [frequency_map.get(v, 0.0) for v in values]
    
    return encoded


def binary_encode_country(values: List[str]) -> List[int]:
    """
    Binary encoding for country: US = 1, Non-US = 0.
    
    Args:
        values: List of country values
        feature_name: Name of the feature
    
    Returns:
        List of binary values
    """
    encoded = [1 if str(v).strip().upper() == 'US' else 0 for v in values]
    return encoded

def manual_onehot_encode(values: List[str], feature_name: str) -> Dict[str, List[int]]:
    """
    Manual one-hot encoding without sklearn.
    
    Args:
        values: List of categorical values
        feature_name: Name of the feature
    
    Returns:
        Dictionary mapping category names to binary column lists
    """
    # Get unique categories
    unique_cats = sorted(set([str(v).strip() if v else 'missing' for v in values]))
    
    # Create one-hot encoding
    encoding = {}
    for cat in unique_cats:
        encoding[f'{feature_name}_{cat}'] = [1 if str(v).strip() == cat else 0 for v in values]
    
    return encoding


def encode_categorical_feature(data: List[Dict], fieldname: str, use_sklearn: bool = True) -> Dict[str, List]:
    """
    One-hot encode a single categorical feature.
    
    Args:
        data: List of record dictionaries
        fieldname: Name of the categorical feature
        use_sklearn: Whether to use sklearn's OneHotEncoder
    
    Returns:
        Dictionary mapping new column names to value lists
    """
    # Extract values
    values = [row.get(fieldname, '') for row in data]
    
    # Handle missing values
    values = [str(v).strip() if v and str(v).strip() else 'missing' for v in values]
    
    if use_sklearn and SKLEARN_AVAILABLE:
        # Use sklearn OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        values_2d = [[v] for v in values]
        encoded = encoder.fit_transform(values_2d)
        
        # Get feature names
        feature_names = encoder.get_feature_names_out([fieldname])
        
        # Convert to dictionary
        result = {}
        for i, name in enumerate(feature_names):
            # Clean feature name (remove special characters)
            clean_name = name.replace(f'{fieldname}_', '').replace(' ', '_').replace('-', '_')
            result[f'{fieldname}_{clean_name}'] = encoded[:, i].tolist()
        
        return result
    else:
        # Manual encoding
        return manual_onehot_encode(values, fieldname)


def process_csv(input_file: str, output_file: str):
    """
    Process CSV file and perform one-hot encoding on categorical features.
    
    Args:
        input_file: Path to input CSV file with hybrid features
        output_file: Path to output CSV file with one-hot encoded features
    """
    print("=" * 80)
    print("One-Hot Encoding for Categorical Features")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Read input CSV
    print("\nReading data file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
    
    print(f"   Read {len(data):,} records")
    print(f"   Original columns: {len(fieldnames)}")
    
    # Identify which categorical features actually exist in the file
    missing_categorical = [f for f in CATEGORICAL_FEATURES if f not in fieldnames]
    
    if missing_categorical:
        print(f"\n⚠️  Warning: Some categorical features not found: {missing_categorical}")
    
    # Categorize features by encoding strategy
    onehot_to_encode = [f for f in ONEHOT_FEATURES if f in fieldnames]
    frequency_to_encode = [f for f in FREQUENCY_FEATURES if f in fieldnames]
    binary_to_encode = [f for f in BINARY_FEATURES if f in fieldnames]
    drop_features = [f for f in DROP_FEATURES if f in fieldnames]
    
    print("\n📊 Categorical features encoding strategy:")
    print(f"   One-Hot Encoding ({len(onehot_to_encode)}): {', '.join(onehot_to_encode)}")
    print(f"   Frequency Encoding ({len(frequency_to_encode)}): {', '.join(frequency_to_encode)}")
    print(f"   Binary Encoding ({len(binary_to_encode)}): {', '.join(binary_to_encode)}")
    print(f"   Drop ({len(drop_features)}): {', '.join(drop_features)}")
    
    for feat in onehot_to_encode:
        unique_vals = set([row.get(feat, '') for row in data if row.get(feat, '').strip()])
        print(f"      • {feat}: {len(unique_vals)} unique categories")
    
    # Perform encoding with different strategies
    print("\n🔧 Performing optimized encoding...")
    encoded_features = {}
    
    # 1. One-Hot Encoding (low cardinality)
    for feat in onehot_to_encode:
        print(f"   One-Hot encoding {feat}...")
        encoded = encode_categorical_feature(data, feat, use_sklearn=SKLEARN_AVAILABLE)
        encoded_features.update(encoded)
        print(f"      Created {len(encoded)} binary columns")
    
    # 2. Frequency Encoding (high cardinality)
    for feat in frequency_to_encode:
        print(f"   Frequency encoding {feat}...")
        values = [row.get(feat, '') for row in data]
        
        # Normalize state names if it's the state feature
        if feat == 'state':
            values = [normalize_state(v) for v in values]
        else:
            values = [str(v).strip() if v and str(v).strip() else 'missing' for v in values]
        
        # Frequency encode
        encoded_values = frequency_encode(values)
        encoded_features[f'{feat}_frequency'] = encoded_values
        
        unique_vals = len(set(values))
        print(f"      Created 1 column (from {unique_vals} unique categories)")
    
    # 3. Binary Encoding
    for feat in binary_to_encode:
        print(f"   Binary encoding {feat}...")
        values = [row.get(feat, '') for row in data]
        encoded_values = binary_encode_country(values)
        encoded_features[f'{feat}_binary'] = encoded_values
        print("      Created 1 binary column")
    
    # 4. Drop features (just log, no encoding)
    if drop_features:
        print(f"   Dropping features: {', '.join(drop_features)}")
    
    # Identify columns to keep
    print("\n📋 Identifying columns to keep...")
    
    # Columns to keep:
    # 1. All numeric features (NLP features, numeric columns)
    # 2. Target variable and important indicators
    # 3. Encoded features (already created)
    # 4. Exclude: original categorical features (now encoded)
    # 5. Exclude: text fields (already converted to NLP features)
    
    all_categorical = onehot_to_encode + frequency_to_encode + binary_to_encode + drop_features
    
    columns_to_keep = []
    
    for col in fieldnames:
        # Always keep target and important indicators
        if col in FEATURES_TO_KEEP:
            columns_to_keep.append(col)
            continue
        
        # Skip original categorical features (will be replaced by encoded versions)
        if col in all_categorical:
            continue
        
        # Skip text fields (already converted to NLP)
        if col in TEXT_FIELDS_TO_DROP:
            continue
        
        # Keep everything else (NLP features, numeric features, etc.)
        columns_to_keep.append(col)
    
    print(f"   Keeping {len(columns_to_keep)} original columns")
    print(f"   Adding {len(encoded_features)} encoded columns")
    
    # Create output fieldnames
    output_fieldnames = columns_to_keep + list(encoded_features.keys())
    
    print(f"\n📊 Final feature count: {len(output_fieldnames)}")
    print(f"   Original features kept: {len(columns_to_keep)}")
    print(f"   Encoded features: {len(encoded_features)}")
    print(f"      - One-Hot: {sum(1 for k in encoded_features.keys() if any(k.startswith(f'{f}_') for f in onehot_to_encode))}")
    print(f"      - Frequency: {sum(1 for k in encoded_features.keys() if '_frequency' in k)}")
    print(f"      - Binary: {sum(1 for k in encoded_features.keys() if '_binary' in k)}")
    
    # Create output records
    print("\n🔄 Creating output records...")
    processed_records = []
    
    for idx, record in enumerate(data):
        new_record = {}
        
        # Add kept columns
        for col in columns_to_keep:
            new_record[col] = record.get(col, '')
        
        # Add encoded features
        for col_name, values in encoded_features.items():
            new_record[col_name] = values[idx]
        
        processed_records.append(new_record)
        
        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"   Processed: {idx + 1:,}/{len(data):,} records")
    
    # Write output CSV
    print("\n💾 Writing output file...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_records)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Input records: {len(data):,}")
    print(f"✅ Output records: {len(processed_records):,}")
    print(f"✅ Input columns: {len(fieldnames)}")
    print(f"✅ Output columns: {len(output_fieldnames)}")
    print(f"✅ Columns added: {len(encoded_features)} (optimized encoding)")
    print(f"✅ Columns removed: {len(all_categorical) + len([c for c in TEXT_FIELDS_TO_DROP if c in fieldnames])} (categorical + text fields)")
    print("\n📈 Optimization Summary:")
    print(f"   Before: {len(onehot_to_encode) + len(frequency_to_encode) + len(binary_to_encode)} categorical features")
    print(f"   After: {len(encoded_features)} encoded features")
    print(f"   Reduction: {len(onehot_to_encode) + len(frequency_to_encode) + len(binary_to_encode) - len(encoded_features)} fewer columns")
    print(f"\n📁 Output file: {output_file}")
    print("\n💡 The dataset is now ready for Linear Regression, XGBoost and Random Forest!")


def main():
    """Main entry point."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
    
    # Default file paths
    input_file = os.path.join(project_root, 'salary_data_with_w2v_features.csv')
    output_file = os.path.join(project_root, 'salary_data_with_w2v_onehot_features.csv')
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        process_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {input_file}")
        print("   Please run hybrid.py first to generate features.")
        sys.exit(1)
    except (OSError, KeyError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

