"""
Text Preprocessing Module
=========================

This module provides functions for cleaning and preprocessing text data
for NLP feature extraction.

Functions:
- clean_text: Clean and normalize text
- tokenize_text: Split text into tokens
- preprocess_field: Preprocess text fields based on their type
"""

import re
from typing import List


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fields that use '|' as separator
PIPE_SEPARATED_FIELDS = ['all_skills', 'all_industries', 'benefits_list']


# =============================================================================
# TEXT PREPROCESSING FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text for NLP processing.
    
    Steps:
    1. Convert to lowercase
    2. Preserve important technical terms (C++, C#, .NET, etc.)
    3. Remove URLs
    4. Remove email addresses
    5. Remove special characters (keep alphanumeric and spaces)
    6. Remove extra whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
        
    Example:
        Input:  "Senior Python Developer (C++) - Apply at http://jobs.com"
        Output: "senior python developer cplusplus apply at"
    """
    if not text or text.strip() == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Preserve important technical terms
    text = text.replace('c++', 'cplusplus')
    text = text.replace('c#', 'csharp')
    text = text.replace('.net', 'dotnet')
    text = text.replace('node.js', 'nodejs')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Simple tokenizer that splits on whitespace and filters empty strings.
    For more advanced tokenization, consider using NLTK or spaCy.
    
    Args:
        text: Cleaned text string
        
    Returns:
        List of tokens (words)
        
    Example:
        Input:  "senior python developer"
        Output: ["senior", "python", "developer"]
    """
    if not text:
        return []
    
    # Split on whitespace and filter empty strings
    tokens = [token for token in text.split() if token.strip()]
    return tokens


def preprocess_field(text: str, field_name: str) -> List[str]:
    """
    Preprocess a text field based on its type.
    
    For fields like all_skills, all_industries, benefits_list that use '|' separator,
    we split by '|' first, then clean and tokenize each item.
    For regular text fields, we clean then tokenize.
    
    Args:
        text: Raw text from CSV
        field_name: Name of the field
        
    Returns:
        List of tokens
        
    Example:
        Input:  text="Python|Java|C++", field_name="all_skills"
        Output: ["python", "java", "cplusplus"]
        
        Input:  text="Senior Software Engineer", field_name="title"
        Output: ["senior", "software", "engineer"]
    """
    if not text or text.strip() == '':
        return []
    
    if field_name in PIPE_SEPARATED_FIELDS:
        # Split by '|' first, then clean and tokenize each item
        items = [item.strip() for item in text.split('|') if item.strip()]
        all_tokens = []
        for item in items:
            cleaned = clean_text(item)
            tokens = tokenize_text(cleaned)
            all_tokens.extend(tokens)
        return all_tokens
    else:
        # Regular text fields: clean then tokenize
        cleaned = clean_text(text)
        return tokenize_text(cleaned)

