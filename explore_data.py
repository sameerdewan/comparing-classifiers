#!/usr/bin/env python3
"""
Generic CSV Data Exploration Script

Usage:
    python explore_data.py <csv_path> [--sep <delimiter>]
    
Output:
    Comprehensive data profile printed to stdout (redirect to file for reference)
    
Example:
    python explore_data.py data/mydata.csv > data_profile.txt
    python explore_data.py data/mydata.csv --sep ";" > data_profile.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def detect_separator(file_path: str, num_lines: int = 5) -> str:
    """Auto-detect CSV separator by testing common delimiters."""
    common_seps = [',', ';', '\t', '|']
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sample_lines = [f.readline() for _ in range(num_lines)]
    
    sample_text = ''.join(sample_lines)
    
    # Count occurrences and consistency of each separator
    best_sep = ','
    best_score = 0
    
    for sep in common_seps:
        counts = [line.count(sep) for line in sample_lines if line.strip()]
        if not counts:
            continue
        # Good separator has consistent count across lines and count > 0
        if min(counts) > 0 and max(counts) == min(counts):
            score = min(counts)
            if score > best_score:
                best_score = score
                best_sep = sep
    
    return best_sep


def detect_placeholder_values(series: pd.Series) -> dict:
    """Detect common placeholder values that represent missing data."""
    placeholders = ['unknown', 'Unknown', 'UNKNOWN', 'na', 'NA', 'N/A', 'n/a', 
                    'null', 'NULL', 'Null', 'none', 'None', 'NONE', '?', '-', 
                    '.', '', ' ', 'missing', 'Missing', 'MISSING', 'nan', 'NaN']
    
    found = {}
    if series.dtype == 'object':
        value_counts = series.value_counts()
        for placeholder in placeholders:
            if placeholder in value_counts.index:
                found[placeholder] = int(value_counts[placeholder])
    
    return found


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}\n")


def analyze_numeric_column(series: pd.Series) -> dict:
    """Get statistics for a numeric column."""
    return {
        'count': int(series.count()),
        'null_count': int(series.isnull().sum()),
        'null_pct': float(series.isnull().sum() / len(series) * 100),
        'mean': float(series.mean()) if not series.isnull().all() else None,
        'std': float(series.std()) if not series.isnull().all() else None,
        'min': float(series.min()) if not series.isnull().all() else None,
        'q25': float(series.quantile(0.25)) if not series.isnull().all() else None,
        'median': float(series.median()) if not series.isnull().all() else None,
        'q75': float(series.quantile(0.75)) if not series.isnull().all() else None,
        'max': float(series.max()) if not series.isnull().all() else None,
    }


def analyze_categorical_column(series: pd.Series) -> dict:
    """Get statistics for a categorical column."""
    value_counts = series.value_counts()
    top_n = min(10, len(value_counts))
    
    return {
        'count': int(series.count()),
        'null_count': int(series.isnull().sum()),
        'null_pct': float(series.isnull().sum() / len(series) * 100),
        'unique_count': int(series.nunique()),
        'top_values': value_counts.head(top_n).to_dict(),
        'top_values_pct': (value_counts.head(top_n) / len(series) * 100).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description='Generic CSV Data Exploration Tool')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('--sep', default=None, help='CSV separator (auto-detected if not provided)')
    args = parser.parse_args()
    
    csv_path = args.csv_path
    
    # Validate file exists
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    # Detect or use provided separator
    if args.sep:
        separator = args.sep
        print(f"Using provided separator: '{separator}'")
    else:
        separator = detect_separator(csv_path)
        print(f"Auto-detected separator: '{separator}'")
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, sep=separator)
    
    # =========================================================================
    # SECTION 1: BASIC INFO
    # =========================================================================
    print_section("1. BASIC INFORMATION")
    
    print(f"File: {csv_path}")
    print(f"Separator: '{separator}'")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"\nColumn names ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2}. {col}")
    
    # =========================================================================
    # SECTION 2: DATA TYPES
    # =========================================================================
    print_section("2. DATA TYPES")
    
    dtype_counts = df.dtypes.value_counts()
    print("Data type summary:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    print("\nColumn data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Identify numeric vs categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # =========================================================================
    # SECTION 3: MISSING VALUES
    # =========================================================================
    print_section("3. MISSING VALUES ANALYSIS")
    
    # Standard null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    print(f"Total null values: {total_nulls}")
    if total_nulls > 0:
        print("\nNull counts per column:")
        for col in df.columns:
            if null_counts[col] > 0:
                pct = null_counts[col] / len(df) * 100
                print(f"  {col}: {null_counts[col]} ({pct:.2f}%)")
    else:
        print("No standard null values found.")
    
    # Placeholder values
    print("\nPlaceholder value detection (potential missing data):")
    placeholder_found = False
    for col in categorical_cols:
        placeholders = detect_placeholder_values(df[col])
        if placeholders:
            placeholder_found = True
            print(f"\n  {col}:")
            for placeholder, count in placeholders.items():
                pct = count / len(df) * 100
                print(f"    '{placeholder}': {count} ({pct:.2f}%)")
    
    if not placeholder_found:
        print("  No common placeholder values detected.")
    
    # =========================================================================
    # SECTION 4: NUMERIC COLUMN STATISTICS
    # =========================================================================
    print_section("4. NUMERIC COLUMN STATISTICS")
    
    if numeric_cols:
        for col in numeric_cols:
            stats = analyze_numeric_column(df[col])
            print(f"\n{col}:")
            print(f"  Count: {stats['count']}, Nulls: {stats['null_count']} ({stats['null_pct']:.2f}%)")
            if stats['mean'] is not None:
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                print(f"  Quartiles: Q25={stats['q25']:.4f}, Median={stats['median']:.4f}, Q75={stats['q75']:.4f}")
    else:
        print("No numeric columns found.")
    
    # =========================================================================
    # SECTION 5: CATEGORICAL COLUMN STATISTICS
    # =========================================================================
    print_section("5. CATEGORICAL COLUMN STATISTICS")
    
    if categorical_cols:
        for col in categorical_cols:
            stats = analyze_categorical_column(df[col])
            print(f"\n{col}:")
            print(f"  Count: {stats['count']}, Nulls: {stats['null_count']} ({stats['null_pct']:.2f}%)")
            print(f"  Unique values: {stats['unique_count']}")
            print(f"  Top values:")
            for val, count in stats['top_values'].items():
                pct = stats['top_values_pct'][val]
                print(f"    '{val}': {count} ({pct:.2f}%)")
    else:
        print("No categorical columns found.")
    
    # =========================================================================
    # SECTION 6: POTENTIAL TARGET COLUMNS
    # =========================================================================
    print_section("6. POTENTIAL TARGET COLUMNS (Low Cardinality: 2-10 unique values)")
    
    potential_targets = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 10:
            potential_targets.append((col, unique_count))
    
    if potential_targets:
        for col, unique_count in potential_targets:
            print(f"\n{col} ({unique_count} unique values):")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                print(f"  '{val}': {count} ({pct:.2f}%)")
    else:
        print("No columns with 2-10 unique values found.")
    
    # =========================================================================
    # SECTION 7: CORRELATION MATRIX (Numeric columns)
    # =========================================================================
    print_section("7. CORRELATION MATRIX (Numeric Columns)")
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        print("Correlation matrix (showing correlations > 0.5 or < -0.5):\n")
        
        high_corr_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.5:
                        high_corr_pairs.append((col1, col2, corr))
        
        if high_corr_pairs:
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} <-> {col2}: {corr:.4f}")
        else:
            print("  No strong correlations (|r| > 0.5) found.")
        
        print("\nFull correlation matrix:")
        print(corr_matrix.round(3).to_string())
    else:
        print("Not enough numeric columns for correlation analysis.")
    
    # =========================================================================
    # SECTION 8: SAMPLE ROWS
    # =========================================================================
    print_section("8. SAMPLE ROWS")
    
    print("First 5 rows:")
    print(df.head().to_string())
    
    print("\nLast 5 rows:")
    print(df.tail().to_string())
    
    # =========================================================================
    # SECTION 9: SUMMARY FOR ML PIPELINE
    # =========================================================================
    print_section("9. SUMMARY FOR ML PIPELINE")
    
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    if potential_targets:
        print(f"\nLikely target column(s):")
        for col, unique_count in potential_targets:
            if unique_count == 2:
                print(f"  {col} (binary classification)")
            else:
                print(f"  {col} (multi-class: {unique_count} classes)")
    
    print("\n" + "=" * 80)
    print(" END OF DATA PROFILE")
    print("=" * 80)


if __name__ == "__main__":
    main()

