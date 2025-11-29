"""
Optimized Feature Encoding with Ordinal Support
===============================================

This script performs optimized encoding on categorical features, specifically adding
Ordinal Encoding support for 'formatted_experience_level'.

Input:  salary_data_with_nlp_hybrid_features.csv
Output: salary_data_final_ordinal.csv

Encoding Strategies:
- Ordinal Encoding: formatted_experience_level (0-5 scale)
- One-Hot Encoding: formatted_work_type, company_size (low cardinality)
- Frequency Encoding: city, state (high cardinality)
- Binary Encoding: country (US vs Non-US)
- Drop: currency

Mappings:
formatted_experience_level:
  Internship -> 0
  Entry level -> 1
  Associate -> 2
  Mid-Senior level -> 3
  Director -> 4
  Executive -> 5
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

# 1. Features for Ordinal Encoding (Ordered categories)
ORDINAL_FEATURES = {
    'formatted_experience_level': {
        'Internship': 0,
        'Entry level': 1,
        'Associate': 2,
        'Mid-Senior level': 3,
        'Director': 4,
        'Executive': 5,
        # Handle potential missing or other values
        '': -1,
        'missing': -1
    }
}

# 2. Features for One-Hot Encoding (Low cardinality, nominal)
ONEHOT_FEATURES = [
    'formatted_work_type',
    'company_size'  # Keeping as One-Hot for now as it's categorical in dictionary
]

# 3. Features for Frequency Encoding (High cardinality)
FREQUENCY_FEATURES = [
    'city',
    'state'
]

# 4. Features for Binary Encoding
BINARY_FEATURES = [
    'country'  # US vs Non-US
]

# 5. Features to Drop
DROP_FEATURES = [
    'currency'
]

# State normalization map
STATE_NORMALIZATION = {
    'California': 'CA', 'New York': 'NY', 'Texas': 'TX', 'Illinois': 'IL',
    'Florida': 'FL', 'Pennsylvania': 'PA', 'Ohio': 'OH', 'Georgia': 'GA',
    'North Carolina': 'NC', 'Michigan': 'MI', 'New Jersey': 'NJ', 'Virginia': 'VA',
    'Washington': 'WA', 'Arizona': 'AZ', 'Massachusetts': 'MA', 'Tennessee': 'TN',
    'Indiana': 'IN', 'Missouri': 'MO', 'Maryland': 'MD', 'Wisconsin': 'WI',
    'Colorado': 'CO', 'Minnesota': 'MN', 'South Carolina': 'SC', 'Alabama': 'AL',
    'Louisiana': 'LA', 'Kentucky': 'KY', 'Oregon': 'OR', 'Oklahoma': 'OK',
    'Connecticut': 'CT', 'Utah': 'UT', 'Iowa': 'IA', 'Nevada': 'NV',
    'Arkansas': 'AR', 'Mississippi': 'MS', 'Kansas': 'KS', 'New Mexico': 'NM',
    'Nebraska': 'NE', 'West Virginia': 'WV', 'Idaho': 'ID', 'Hawaii': 'HI',
    'New Hampshire': 'NH', 'Maine': 'ME', 'Montana': 'MT', 'Rhode Island': 'RI',
    'Delaware': 'DE', 'South Dakota': 'SD', 'North Dakota': 'ND', 'Alaska': 'AK',
    'Vermont': 'VT', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

# Text fields to drop (already converted to NLP features)
TEXT_FIELDS_TO_DROP = [
    'company_name', 'title', 'description', 'location',
    'description_company', 'primary_industry', 'all_industries',
    'primary_skill', 'all_skills', 'benefits_list',
    'primary_company_industry', 'all_company_industries',
    'primary_speciality', 'all_specialities'
]

FEATURES_TO_KEEP = ['salary_normalized', 'benefits_list_missing']

# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def normalize_state(state: str) -> str:
    if not state or state.strip() == '' or state == '0':
        return 'missing'
    state = state.strip()
    if len(state) == 2 and state.isalpha():
        return state.upper()
    return STATE_NORMALIZATION.get(state, state)

def frequency_encode(values: List[str]) -> List[float]:
    counter = Counter(values)
    total = len(values)
    frequency_map = {}
    for category, count in counter.items():
        frequency_map[category] = math.log(count + 1) / math.log(total + 1)
    return [frequency_map.get(v, 0.0) for v in values]

def binary_encode_country(values: List[str]) -> List[int]:
    return [1 if str(v).strip().upper() == 'US' else 0 for v in values]

def ordinal_encode_feature(data: List[Dict], fieldname: str, mapping: Dict) -> List[int]:
    """Perform ordinal encoding based on provided mapping."""
    values = [row.get(fieldname, '') for row in data]
    encoded = []
    for v in values:
        v_str = str(v).strip()
        if v_str in mapping:
            encoded.append(mapping[v_str])
        else:
            # Try finding a matching key if minor differences exist
            found = False
            for key in mapping:
                if key in v_str:
                    encoded.append(mapping[key])
                    found = True
                    break
            if not found:
                encoded.append(-1) # Default fallback
    return encoded

def encode_onehot_feature(data: List[Dict], fieldname: str) -> Dict[str, List]:
    values = [str(row.get(fieldname, '')).strip() or 'missing' for row in data]
    
    if SKLEARN_AVAILABLE:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        values_2d = [[v] for v in values]
        encoded = encoder.fit_transform(values_2d)
        feature_names = encoder.get_feature_names_out([fieldname])
        
        result = {}
        for i, name in enumerate(feature_names):
            clean_name = name.replace(f'{fieldname}_', '').replace(' ', '_').replace('-', '_')
            result[f'{fieldname}_{clean_name}'] = encoded[:, i].tolist()
        return result
    else:
        # Manual fallback
        unique_cats = sorted(set(values))
        encoding = {}
        for cat in unique_cats:
            col_name = f"{fieldname}_{cat.replace(' ', '_')}"
            encoding[col_name] = [1 if v == cat else 0 for v in values]
        return encoding

# =============================================================================
# MAIN PROCESS
# =============================================================================

def process_csv(input_file: str, output_file: str):
    print("=" * 80)
    print("FEATURE ENCODING WITH ORDINAL SUPPORT")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Read Data
    print("\nReading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
    print(f"   Read {len(data):,} records")

    encoded_features = {}

    # 1. Ordinal Encoding
    print("\n🔧 Applying Ordinal Encoding...")
    for feat, mapping in ORDINAL_FEATURES.items():
        if feat in fieldnames:
            print(f"   Encoding {feat} (0-5 scale)...")
            encoded_values = ordinal_encode_feature(data, feat, mapping)
            encoded_features[f'{feat}_ordinal'] = encoded_values
            
            # Verify distribution
            counts = Counter(encoded_values)
            print(f"      Distribution: {dict(sorted(counts.items()))}")
        else:
            print(f"   ⚠️ Feature {feat} not found!")

    # 2. One-Hot Encoding
    print("\n🔧 Applying One-Hot Encoding...")
    for feat in ONEHOT_FEATURES:
        if feat in fieldnames:
            print(f"   Encoding {feat}...")
            encoded = encode_onehot_feature(data, feat)
            encoded_features.update(encoded)
            print(f"      Created {len(encoded)} columns")

    # 3. Frequency Encoding
    print("\n🔧 Applying Frequency Encoding...")
    for feat in FREQUENCY_FEATURES:
        if feat in fieldnames:
            print(f"   Encoding {feat}...")
            values = [row.get(feat, '') for row in data]
            if feat == 'state':
                values = [normalize_state(v) for v in values]
            else:
                values = [str(v).strip() if v and str(v).strip() else 'missing' for v in values]
            
            encoded_features[f'{feat}_frequency'] = frequency_encode(values)
            print("      Created 1 column")

    # 4. Binary Encoding
    print("\n🔧 Applying Binary Encoding...")
    for feat in BINARY_FEATURES:
        if feat in fieldnames:
            print(f"   Encoding {feat}...")
            values = [row.get(feat, '') for row in data]
            encoded_features[f'{feat}_binary'] = binary_encode_country(values)
            print("      Created 1 column")

    # Build Output Columns
    print("\n📋 Assembling final dataset...")
    all_categorical = list(ORDINAL_FEATURES.keys()) + ONEHOT_FEATURES + FREQUENCY_FEATURES + BINARY_FEATURES + DROP_FEATURES
    
    columns_to_keep = []
    for col in fieldnames:
        if col in FEATURES_TO_KEEP:
            columns_to_keep.append(col)
            continue
        if col in all_categorical:
            continue
        if col in TEXT_FIELDS_TO_DROP:
            continue
        columns_to_keep.append(col)

    output_fieldnames = columns_to_keep + list(encoded_features.keys())
    print(f"   Original columns kept: {len(columns_to_keep)}")
    print(f"   New encoded columns:   {len(encoded_features)}")
    print(f"   Total features:        {len(output_fieldnames)}")

    # Write Output
    print("\n💾 Writing output file...")
    processed_records = []
    for idx, record in enumerate(data):
        new_record = {}
        for col in columns_to_keep:
            new_record[col] = record.get(col, '')
        for col, values in encoded_features.items():
            new_record[col] = values[idx]
        processed_records.append(new_record)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_records)

    print(f"\n✅ Done! Saved to: {output_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
    
    input_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_features.csv')
    output_file = os.path.join(project_root, 'salary_data_ready_after_nlp_and_feature_encoding.csv')
    
    if len(sys.argv) > 1: input_file = sys.argv[1]
    if len(sys.argv) > 2: output_file = sys.argv[2]

    try:
        process_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"\n❌ Error: Input file not found: {input_file}")
        print("   Please run hybrid.py first.")

if __name__ == '__main__':
    main()

