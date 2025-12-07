#!/usr/bin/env python3
"""
Bank Marketing Classifier Comparison - Solution Script

This script solves all problems from prompt_III.ipynb.
Each problem's output is saved to outputs/ directory for reference.
ALL FINDINGS ARE DERIVED FROM ACTUAL DATA - NO HARDCODED INTERPRETATIONS.

Usage: python solution.py
"""

import time
import warnings
import os
import json
from io import StringIO

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score, recall_score)

os.makedirs('outputs', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)


def save_output(filename: str, content: str) -> None:
    """Save content to outputs directory."""
    filepath = f"outputs/{filename}"
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  -> Saved: {filepath}")


def print_section(title: str) -> None:
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


# =============================================================================
# PROBLEM 1: Understanding the Data
# =============================================================================
def problem_1(df):
    """How many marketing campaigns does this data represent?"""
    print_section("PROBLEM 1: Understanding the Data")
    
    # Extract actual months from data
    months_in_data = df['month'].unique().tolist()
    month_counts = df['month'].value_counts()
    
    # Calculate timespan indicators
    n_unique_months = len(months_in_data)
    total_contacts = len(df)
    
    output = f"""PROBLEM 1: Understanding the Data
=====================================

QUESTION: How many marketing campaigns does this data represent?

DATA-DERIVED FINDINGS:
- Total contacts in dataset: {total_contacts:,}
- Unique months in data: {n_unique_months}
- Months present: {sorted(months_in_data)}

MONTH DISTRIBUTION:
{month_counts.to_string()}

EXTERNAL SOURCE (CRISP-DM-BANK.pdf):
- The paper states: 17 campaigns conducted
- Timeframe: May 2008 to November 2010 (~2.5 years)
- {n_unique_months} unique months in data across {total_contacts:,} contacts
- Average contacts per campaign: {total_contacts / 17:,.0f}

ANSWER: 17 marketing campaigns (from paper documentation)
"""
    print(output)
    save_output("problem_1_campaigns.txt", output)
    
    return {'n_campaigns': 17, 'months': months_in_data, 'total_contacts': total_contacts}


# =============================================================================
# PROBLEM 2: Read in the Data
# =============================================================================
def problem_2():
    """Read in the dataset and verify its structure."""
    print_section("PROBLEM 2: Read in the Data")
    
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    
    # Derive findings from actual data
    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    output = f"""PROBLEM 2: Read in the Data
=====================================

CODE:
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')

DATA SHAPE:
- Rows: {n_rows:,}
- Columns: {n_cols}
- Memory: {memory_mb:.2f} MB

COLUMN TYPES FOUND:
- Numeric columns ({len(numeric_cols)}): {numeric_cols}
- Categorical columns ({len(categorical_cols)}): {categorical_cols}

ALL COLUMNS:
{chr(10).join(f'  {i+1}. {col} ({df[col].dtype})' for i, col in enumerate(df.columns))}

SAMPLE DATA (first 3 rows):
{df.head(3).to_string()}

PATH NOTE: Correct path is 'data/bank-additional-full.csv'
"""
    print(output)
    save_output("problem_2_data_loaded.txt", output)
    
    return df


# =============================================================================
# PROBLEM 3: Understanding the Features
# =============================================================================
def problem_3(df):
    """Analyze features for missing values and data types."""
    print_section("PROBLEM 3: Understanding the Features")
    
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Analyze placeholder values
    categorical_cols = df.select_dtypes(include=['object']).columns
    unknown_analysis = {}
    for col in categorical_cols:
        unknown_count = int((df[col] == 'unknown').sum())
        if unknown_count > 0:
            pct = unknown_count / len(df) * 100
            unknown_analysis[col] = {'count': unknown_count, 'pct': round(pct, 2)}
    
    # Check for actual nulls
    null_total = df.isnull().sum().sum()
    
    # Find column with most unknowns
    if unknown_analysis:
        worst_col = max(unknown_analysis.items(), key=lambda x: x[1]['pct'])
        worst_col_name, worst_col_data = worst_col
    else:
        worst_col_name, worst_col_data = None, None
    
    # Derive observations from data
    observations = []
    if null_total == 0:
        observations.append(f"No standard null values found (total nulls: {null_total})")
    else:
        observations.append(f"Found {null_total} null values")
    
    if unknown_analysis:
        observations.append(f"'unknown' placeholder found in {len(unknown_analysis)} columns")
        if worst_col_data and worst_col_data['pct'] > 10:
            observations.append(f"WARNING: {worst_col_name} has {worst_col_data['pct']}% unknowns - this is significant (>10%)")
        elif worst_col_data:
            observations.append(f"Column with most unknowns: {worst_col_name} ({worst_col_data['pct']}%)")
    
    output = f"""PROBLEM 3: Understanding the Features
=====================================

DATA INFO:
{info_str}

NULL VALUE CHECK:
- Standard nulls (NaN): {null_total}

PLACEHOLDER 'unknown' VALUES:
{json.dumps(unknown_analysis, indent=2) if unknown_analysis else 'None found'}

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}

TYPE COERCION NEEDED: {'No' if null_total == 0 else 'Yes - handle nulls'}
DECISION: {'Keep unknown as category - it may indicate client declined to answer' if unknown_analysis else 'No special handling needed'}
"""
    print(output)
    save_output("problem_3_features.txt", output)
    
    return df, unknown_analysis


# =============================================================================
# PROBLEM 4: Understanding the Task
# =============================================================================
def problem_4(df):
    """State the business objective based on data analysis."""
    print_section("PROBLEM 4: Understanding the Task")
    
    # Analyze target column
    target_counts = df['y'].value_counts()
    target_pcts = df['y'].value_counts(normalize=True) * 100
    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    majority_pct = target_pcts[majority_class]
    minority_pct = target_pcts[minority_class]
    imbalance_ratio = target_counts[majority_class] / target_counts[minority_class]
    
    # Derive observations based on actual numbers
    observations = []
    if imbalance_ratio > 5:
        observations.append(f"SEVERE imbalance detected: {imbalance_ratio:.1f}:1 ratio")
        observations.append("Accuracy will be misleading - a model predicting all 'no' gets {:.1f}% accuracy".format(majority_pct))
        observations.append("Must use metrics like F1, precision, recall for minority class")
    elif imbalance_ratio > 2:
        observations.append(f"Moderate imbalance: {imbalance_ratio:.1f}:1 ratio")
        observations.append("Consider stratified sampling and balanced metrics")
    else:
        observations.append(f"Balanced classes: {imbalance_ratio:.1f}:1 ratio")
        observations.append("Accuracy is a valid metric")
    
    output = f"""PROBLEM 4: Understanding the Task - Business Objective
=====================================

TARGET VARIABLE: 'y'
- Values: {list(target_counts.index)}
- This represents: whether client subscribed to term deposit

TARGET DISTRIBUTION:
{target_counts.to_string()}

{target_pcts.round(2).to_string()}

CLASS ANALYSIS:
- Majority class: '{majority_class}' ({majority_pct:.2f}%)
- Minority class: '{minority_class}' ({minority_pct:.2f}%)
- Imbalance ratio: {imbalance_ratio:.1f}:1

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}

BUSINESS OBJECTIVE:
Predict whether a client will subscribe to a term deposit (y='{minority_class}')
based on client demographics, campaign data, and economic indicators.

METRIC RECOMMENDATION (based on {imbalance_ratio:.1f}:1 imbalance):
{'F1-score or ROC-AUC (accuracy misleading due to imbalance)' if imbalance_ratio > 3 else 'Accuracy acceptable, but also track F1'}
"""
    print(output)
    save_output("problem_4_business_objective.txt", output)
    
    return {
        'target_counts': target_counts.to_dict(),
        'target_pcts': target_pcts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'majority_class': majority_class,
        'minority_class': minority_class
    }


# =============================================================================
# PROBLEM 5: Engineering Features
# =============================================================================
def problem_5(df):
    """Prepare features and target for modeling."""
    print_section("PROBLEM 5: Engineering Features")
    
    df_processed = df.copy()
    
    # Step 1: Identify and drop duration
    has_duration = 'duration' in df_processed.columns
    if has_duration:
        duration_corr_with_target = df_processed.groupby('y')['duration'].mean()
        df_processed = df_processed.drop('duration', axis=1)
    
    # Step 2: Encode target
    target_values = df_processed['y'].unique()
    df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
    
    # Step 3: Separate features and target
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']
    
    # Identify column types from actual data
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    n_features_before = len(X.columns)
    
    # Step 4: One-hot encode
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    n_features_after = len(X_encoded.columns)
    
    # Step 5: Scale numeric features
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    
    output = f"""PROBLEM 5: Engineering Features
=====================================

STEP 1 - DROP DURATION:
- Column 'duration' present: {has_duration}
- Reason: Duration not known before call (data leakage)
{"- Duration by target: " + duration_corr_with_target.to_string() if has_duration else ""}

STEP 2 - ENCODE TARGET:
- Original values: {list(target_values)}
- Encoding: 'yes' -> 1, 'no' -> 0

STEP 3 - COLUMN TYPES IDENTIFIED:
- Numeric ({len(numeric_cols)}): {numeric_cols}
- Categorical ({len(categorical_cols)}): {categorical_cols}

STEP 4 - ONE-HOT ENCODING:
- Features before: {n_features_before}
- Features after: {n_features_after}
- New features created: {n_features_after - n_features_before}

STEP 5 - SCALING:
- Applied StandardScaler to {len(numeric_cols)} numeric columns
- Transforms to mean=0, std=1

FINAL SHAPES:
- X: {X_encoded.shape}
- y: {y.shape}
- y distribution: 0={int((y==0).sum())}, 1={int((y==1).sum())}
"""
    print(output)
    save_output("problem_5_features_engineered.txt", output)
    
    with open('outputs/feature_names.json', 'w') as f:
        json.dump(list(X_encoded.columns), f, indent=2)
    print("  -> Saved: outputs/feature_names.json")
    
    return X_encoded, y, scaler, numeric_cols, categorical_cols


# =============================================================================
# PROBLEM 6: Train/Test Split
# =============================================================================
def problem_6(X, y):
    """Split data into training and test sets."""
    print_section("PROBLEM 6: Train/Test Split")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Verify stratification worked
    train_pct_1 = y_train.mean() * 100
    test_pct_1 = y_test.mean() * 100
    original_pct_1 = y.mean() * 100
    stratification_error = abs(train_pct_1 - original_pct_1)
    
    output = f"""PROBLEM 6: Train/Test Split
=====================================

CODE:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

SPLIT SIZES:
- Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)
- Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)

CLASS DISTRIBUTION CHECK:
- Original: {original_pct_1:.2f}% positive class
- Training: {train_pct_1:.2f}% positive class  
- Test: {test_pct_1:.2f}% positive class

STRATIFICATION VERIFICATION:
- Stratification error: {stratification_error:.4f}%
- Status: {'SUCCESS - distributions match' if stratification_error < 0.1 else 'WARNING - distributions differ'}

TRAINING SET:
- Class 0: {int((y_train==0).sum())}
- Class 1: {int((y_train==1).sum())}

TEST SET:
- Class 0: {int((y_test==0).sum())}
- Class 1: {int((y_test==1).sum())}
"""
    print(output)
    save_output("problem_6_train_test_split.txt", output)
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# PROBLEM 7: A Baseline Model
# =============================================================================
def problem_7(X_train, X_test, y_train, y_test, target_info):
    """Establish baseline performance."""
    print_section("PROBLEM 7: A Baseline Model")
    
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    
    baseline_train_acc = baseline.score(X_train, y_train)
    baseline_test_acc = baseline.score(X_test, y_test)
    
    # What does baseline predict?
    baseline_pred = baseline.predict(X_test[:5])
    
    # Derive insights from actual numbers
    imbalance_ratio = target_info['imbalance_ratio']
    majority_class = target_info['majority_class']
    
    observations = []
    observations.append(f"Baseline always predicts '{majority_class}' (majority class)")
    observations.append(f"This achieves {baseline_test_acc*100:.2f}% accuracy by doing nothing useful")
    
    if baseline_test_acc > 0.8:
        observations.append(f"WARNING: Baseline >{80}% means accuracy is NOT a good metric")
        observations.append(f"A model with 90% accuracy is only {(0.90 - baseline_test_acc)*100:.1f}% better than guessing")
        recommended_metric = "F1-score"
    else:
        observations.append("Baseline is reasonable - accuracy can be used as a metric")
        recommended_metric = "Accuracy"
    
    output = f"""PROBLEM 7: A Baseline Model
=====================================

BASELINE MODEL: DummyClassifier(strategy='most_frequent')

BASELINE PERFORMANCE:
- Training Accuracy: {baseline_train_acc:.4f}
- Test Accuracy: {baseline_test_acc:.4f}

WHAT BASELINE DOES:
- Predicts: {baseline_pred.tolist()} (always {int(baseline_pred[0])})
- This equals always predicting '{majority_class}'

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}

IMBALANCE CONTEXT:
- Class ratio: {imbalance_ratio:.1f}:1
- Majority class %: {baseline_test_acc*100:.2f}%

METRIC RECOMMENDATION: {recommended_metric}
- Any useful model must beat {baseline_test_acc*100:.2f}% baseline
- Better metrics for this data: F1, Precision, Recall, ROC-AUC
"""
    print(output)
    save_output("problem_7_baseline.txt", output)
    
    return baseline, baseline_test_acc


# =============================================================================
# PROBLEM 8: A Simple Model
# =============================================================================
def problem_8(X_train, y_train):
    """Train a Logistic Regression model."""
    print_section("PROBLEM 8: A Simple Model - Logistic Regression")
    
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    output = f"""PROBLEM 8: A Simple Model - Logistic Regression
=====================================

CODE:
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

TRAINING RESULTS:
- Training time: {train_time:.4f} seconds
- Iterations to converge: {lr.n_iter_[0]}
- Converged: {'Yes' if lr.n_iter_[0] < 1000 else 'No - hit max_iter'}
- Number of features used: {lr.coef_.shape[1]}
- Regularization strength (C): {lr.C}
"""
    print(output)
    save_output("problem_8_logistic_regression.txt", output)
    
    return lr, train_time


# =============================================================================
# PROBLEM 9: Score the Model
# =============================================================================
def problem_9(lr, X_train, X_test, y_train, y_test, feature_names, baseline_acc):
    """Score the Logistic Regression model with data-derived insights."""
    print_section("PROBLEM 9: Score the Model")
    
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    y_test_proba = lr.predict_proba(X_test)[:, 1]
    
    # Compute all metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Coefficient analysis
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr.coef_[0]
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    top_10 = coef_df.head(10)
    
    # Data-derived observations
    observations = []
    
    # Accuracy vs baseline
    acc_improvement = test_acc - baseline_acc
    observations.append(f"Accuracy improvement over baseline: +{acc_improvement*100:.2f}%")
    
    # Overfitting check
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.05:
        observations.append(f"POSSIBLE OVERFITTING: Train-Test gap = {overfit_gap*100:.2f}%")
    else:
        observations.append(f"No overfitting detected: Train-Test gap = {overfit_gap*100:.2f}%")
    
    # Recall analysis
    if recall < 0.3:
        observations.append(f"LOW RECALL ({recall:.2f}): Missing {fn} of {fn+tp} actual positives")
    elif recall > 0.7:
        observations.append(f"GOOD RECALL ({recall:.2f}): Catching most positives")
    
    # Precision analysis
    if precision < 0.5:
        observations.append(f"LOW PRECISION ({precision:.2f}): Many false alarms ({fp} false positives)")
    elif precision > 0.7:
        observations.append(f"GOOD PRECISION ({precision:.2f}): Predictions are reliable")
    
    output = f"""PROBLEM 9: Score the Model
=====================================

ACCURACY:
- Training: {train_acc:.4f}
- Test: {test_acc:.4f}
- Baseline: {baseline_acc:.4f}
- Improvement: +{acc_improvement*100:.2f}%

ALL METRICS (Test Set):
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-score: {f1:.4f}
- ROC-AUC: {roc_auc:.4f}

CONFUSION MATRIX:
              Predicted
              No      Yes
Actual No    {tn:5d}   {fp:5d}
Actual Yes   {fn:5d}   {tp:5d}

CONFUSION MATRIX INTERPRETATION:
- True Negatives (correctly rejected): {tn}
- False Positives (false alarms): {fp}
- False Negatives (missed opportunities): {fn}
- True Positives (correctly identified): {tp}

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}

TOP 10 FEATURES (by coefficient magnitude):
{top_10[['feature', 'coefficient']].to_string(index=False)}

COEFFICIENT MEANING:
- Positive = increases subscription probability
- Negative = decreases subscription probability
- Top positive: {top_10[top_10['coefficient'] > 0].iloc[0]['feature'] if len(top_10[top_10['coefficient'] > 0]) > 0 else 'N/A'} ({top_10[top_10['coefficient'] > 0].iloc[0]['coefficient']:.4f})
- Top negative: {top_10[top_10['coefficient'] < 0].iloc[0]['feature'] if len(top_10[top_10['coefficient'] < 0]) > 0 else 'N/A'} ({top_10[top_10['coefficient'] < 0].iloc[0]['coefficient']:.4f})
"""
    print(output)
    save_output("problem_9_model_scores.txt", output)
    
    coef_df.to_csv('outputs/logistic_regression_coefficients.csv', index=False)
    print("  -> Saved: outputs/logistic_regression_coefficients.csv")
    
    return {
        'train_acc': train_acc, 'test_acc': test_acc,
        'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'top_features': top_10[['feature', 'coefficient']].to_dict('records')
    }


# =============================================================================
# PROBLEM 10: Model Comparisons
# =============================================================================
def problem_10(X_train, X_test, y_train, y_test, baseline_acc):
    """Compare multiple classifiers with data-derived insights."""
    print_section("PROBLEM 10: Model Comparisons")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        results.append({
            'Model': name,
            'Train Time (s)': round(train_time, 4),
            'Train Accuracy': round(train_acc, 4),
            'Test Accuracy': round(test_acc, 4),
            'Overfit Gap': round(train_acc - test_acc, 4),
            'vs Baseline': round(test_acc - baseline_acc, 4)
        })
        trained_models[name] = model
        
        print(f"Done! ({train_time:.2f}s)")
    
    results_df = pd.DataFrame(results)
    
    # Data-derived observations
    observations = []
    
    best_test = results_df.loc[results_df['Test Accuracy'].idxmax()]
    worst_test = results_df.loc[results_df['Test Accuracy'].idxmin()]
    fastest = results_df.loc[results_df['Train Time (s)'].idxmin()]
    slowest = results_df.loc[results_df['Train Time (s)'].idxmax()]
    
    observations.append(f"Best test accuracy: {best_test['Model']} ({best_test['Test Accuracy']:.4f})")
    observations.append(f"Worst test accuracy: {worst_test['Model']} ({worst_test['Test Accuracy']:.4f})")
    observations.append(f"Fastest: {fastest['Model']} ({fastest['Train Time (s)']}s)")
    observations.append(f"Slowest: {slowest['Model']} ({slowest['Train Time (s)']}s)")
    
    # Check for overfitting
    for _, row in results_df.iterrows():
        if row['Overfit Gap'] > 0.1:
            observations.append(f"OVERFITTING: {row['Model']} has {row['Overfit Gap']*100:.1f}% train-test gap")
    
    # Check if all beat baseline
    all_beat_baseline = all(r['vs Baseline'] > 0 for _, r in results_df.iterrows())
    if not all_beat_baseline:
        underperformers = results_df[results_df['vs Baseline'] <= 0]['Model'].tolist()
        observations.append(f"BELOW BASELINE: {underperformers}")
    else:
        observations.append("All models beat baseline")
    
    output = f"""PROBLEM 10: Model Comparisons
=====================================

MODELS COMPARED:
- Logistic Regression (max_iter=1000)
- KNN (n_neighbors=5, default)
- Decision Tree (no max_depth, default)
- SVM (RBF kernel, default)

COMPARISON TABLE:
{results_df.to_string(index=False)}

BASELINE REFERENCE: {baseline_acc:.4f}

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}
"""
    print(output)
    save_output("problem_10_model_comparison.txt", output)
    
    results_df.to_csv('outputs/model_comparison_results.csv', index=False)
    print("  -> Saved: outputs/model_comparison_results.csv")
    
    return results_df, trained_models


# =============================================================================
# PROBLEM 11: Improving the Model
# =============================================================================
def problem_11(X_train, X_test, y_train, y_test, baseline_acc, target_info):
    """Hyperparameter tuning with data-derived insights."""
    print_section("PROBLEM 11: Improving the Model")
    
    imbalance_ratio = target_info['imbalance_ratio']
    scoring_metric = 'f1' if imbalance_ratio > 3 else 'accuracy'
    
    print(f"\nScoring metric chosen: {scoring_metric} (based on {imbalance_ratio:.1f}:1 imbalance)")
    print("Cross-validation: 5-fold\n")
    
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]}
        }
    }
    
    tuned_results = []
    best_models = {}
    
    for name, config in param_grids.items():
        print(f"  Tuning {name}...", end=" ", flush=True)
        
        grid_search = GridSearchCV(
            config['model'], config['params'],
            cv=5, scoring=scoring_metric, n_jobs=1, verbose=0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tune_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        y_pred = best_model.predict(X_test)
        
        tuned_results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'CV Score': round(grid_search.best_score_, 4),
            'Test Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Test F1': round(f1_score(y_test, y_pred), 4),
            'Tune Time (s)': round(tune_time, 2)
        })
        
        print(f"Done!")
    
    # SVM with limited tuning
    print("  Tuning SVM (limited)...", end=" ", flush=True)
    svm = SVC(C=1.0, kernel='rbf', random_state=42)
    start_time = time.time()
    svm.fit(X_train, y_train)
    svm_time = time.time() - start_time
    
    y_pred_svm = svm.predict(X_test)
    cv_scores = cross_val_score(svm, X_train, y_train, cv=3, scoring=scoring_metric, n_jobs=1)
    
    tuned_results.append({
        'Model': 'SVM',
        'Best Params': {'C': 1.0, 'kernel': 'rbf'},
        'CV Score': round(cv_scores.mean(), 4),
        'Test Accuracy': round(accuracy_score(y_test, y_pred_svm), 4),
        'Test F1': round(f1_score(y_test, y_pred_svm), 4),
        'Tune Time (s)': round(svm_time, 2)
    })
    best_models['SVM'] = svm
    print(f"Done!")
    
    tuned_df = pd.DataFrame(tuned_results)
    
    # Find best model based on chosen metric
    if scoring_metric == 'f1':
        best_idx = tuned_df['Test F1'].idxmax()
        best_metric_col = 'Test F1'
    else:
        best_idx = tuned_df['Test Accuracy'].idxmax()
        best_metric_col = 'Test Accuracy'
    
    best_model_name = tuned_df.loc[best_idx, 'Model']
    best_model_score = tuned_df.loc[best_idx, best_metric_col]
    
    # Data-derived observations
    observations = []
    observations.append(f"Metric used: {scoring_metric} (due to {imbalance_ratio:.1f}:1 class imbalance)")
    observations.append(f"Best model: {best_model_name} ({best_metric_col}={best_model_score:.4f})")
    
    # Check for improvement
    for _, row in tuned_df.iterrows():
        improvement = row['Test Accuracy'] - baseline_acc
        if improvement > 0.02:
            observations.append(f"{row['Model']}: +{improvement*100:.1f}% vs baseline")
    
    output = f"""PROBLEM 11: Improving the Model
=====================================

METRIC SELECTION:
- Chosen metric: {scoring_metric}
- Reason: Class imbalance ratio is {imbalance_ratio:.1f}:1
{'- F1 balances precision/recall for minority class' if scoring_metric == 'f1' else '- Accuracy is valid for balanced classes'}

HYPERPARAMETER GRIDS SEARCHED:
- Logistic Regression: C=[0.01, 0.1, 1, 10], solver=['lbfgs', 'liblinear']
- KNN: n_neighbors=[3, 5, 7, 9, 11], weights=['uniform', 'distance']
- Decision Tree: max_depth=[3, 5, 7, 10, None], min_samples_split=[2, 5, 10]
- SVM: C=1.0, kernel='rbf' (limited due to training time)

TUNED RESULTS:
"""
    for r in tuned_results:
        output += f"\n{r['Model']}:\n"
        output += f"  Best Params: {r['Best Params']}\n"
        output += f"  CV {scoring_metric.upper()}: {r['CV Score']}\n"
        output += f"  Test Accuracy: {r['Test Accuracy']}\n"
        output += f"  Test F1: {r['Test F1']}\n"
    
    output += f"""
COMPARISON TABLE:
{tuned_df[['Model', 'CV Score', 'Test Accuracy', 'Test F1', 'Tune Time (s)']].to_string(index=False)}

DATA-DERIVED OBSERVATIONS:
{chr(10).join(f'- {obs}' for obs in observations)}

RECOMMENDATION: {best_model_name}
"""
    print(output)
    save_output("problem_11_tuned_models.txt", output)
    
    tuned_df_save = tuned_df.copy()
    tuned_df_save['Best Params'] = tuned_df_save['Best Params'].apply(str)
    tuned_df_save.to_csv('outputs/tuned_model_results.csv', index=False)
    print("  -> Saved: outputs/tuned_model_results.csv")
    
    return tuned_df, best_models, best_model_name


# =============================================================================
# VISUALIZATIONS
# =============================================================================
def create_visualizations(df, results_df, tuned_df, lr, feature_names, baseline_acc):
    """Create and save all visualizations."""
    print_section("CREATING VISUALIZATIONS")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = df['y'].value_counts()
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(target_counts.index, target_counts.values, color=colors)
    ax.set_xlabel('Subscription Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, target_counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/1_target_distribution.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/1_target_distribution.png")
    
    # 2. Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, results_df['Train Accuracy'], width, label='Train', color='#3498db')
    ax1.bar(x + width/2, results_df['Test Accuracy'], width, label='Test', color='#e74c3c')
    ax1.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.3f})')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0.8, 1.0)
    
    ax2 = axes[1]
    ax2.bar(results_df['Model'], results_df['Train Time (s)'], color='#2ecc71')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('visualizations/2_model_comparison.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/2_model_comparison.png")
    
    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': lr.coef_[0]})
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    top_15 = coef_df.nlargest(15, 'abs_coef')
    
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_15['coefficient']]
    ax.barh(top_15['feature'], top_15['coefficient'], color=colors)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Top 15 Features (Logistic Regression)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('visualizations/3_feature_importance.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/3_feature_importance.png")
    
    # 4. Tuned Model F1 Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tuned_df))
    width = 0.35
    ax.bar(x - width/2, tuned_df['CV Score'], width, label='CV Score', color='#3498db')
    ax.bar(x + width/2, tuned_df['Test F1'], width, label='Test F1', color='#e74c3c')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Tuned Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tuned_df['Model'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('visualizations/4_tuned_model_f1.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/4_tuned_model_f1.png")
    
    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/5_correlation_heatmap.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/5_correlation_heatmap.png")
    
    # 6. Subscription by Job
    fig, ax = plt.subplots(figsize=(12, 6))
    job_sub = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values()
    overall_rate = (df['y'] == 'yes').mean() * 100
    colors = ['#2ecc71' if v > overall_rate else '#e74c3c' for v in job_sub.values]
    ax.barh(job_sub.index, job_sub.values, color=colors)
    ax.axvline(x=overall_rate, color='black', linestyle='--', linewidth=2, label=f'Overall: {overall_rate:.1f}%')
    ax.set_xlabel('Subscription Rate (%)', fontsize=12)
    ax.set_title('Subscription Rate by Job Type', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('visualizations/6_subscription_by_job.png', dpi=150)
    plt.close()
    print("  -> Saved: visualizations/6_subscription_by_job.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Run all problems and save outputs."""
    print("=" * 80)
    print(" BANK MARKETING CLASSIFIER COMPARISON")
    print(" All findings derived from actual data analysis")
    print("=" * 80)
    
    # Problem 2 first to get data
    df = problem_2()
    
    # Problem 1 uses df
    campaign_info = problem_1(df)
    
    # Problem 3
    df, unknown_analysis = problem_3(df)
    
    # Problem 4
    target_info = problem_4(df)
    
    # Problem 5
    X, y, scaler, numeric_cols, categorical_cols = problem_5(df)
    
    # Problem 6
    X_train, X_test, y_train, y_test = problem_6(X, y)
    
    # Problem 7
    baseline, baseline_acc = problem_7(X_train, X_test, y_train, y_test, target_info)
    
    # Problem 8
    lr, lr_train_time = problem_8(X_train, y_train)
    
    # Problem 9
    feature_names = X.columns.tolist()
    lr_scores = problem_9(lr, X_train, X_test, y_train, y_test, feature_names, baseline_acc)
    
    # Problem 10
    results_df, models = problem_10(X_train, X_test, y_train, y_test, baseline_acc)
    
    # Problem 11
    tuned_df, best_models, best_model_name = problem_11(X_train, X_test, y_train, y_test, baseline_acc, target_info)
    
    # Visualizations
    create_visualizations(df, results_df, tuned_df, lr, feature_names, baseline_acc)
    
    # Final Summary
    print_section("FINAL SUMMARY")
    
    summary = f"""FINAL SUMMARY
=====================================
All values below are derived from actual data analysis.

DATA:
- Samples: {len(df):,}
- Features: {len(df.columns)}
- Target: 'y' ({target_info['minority_class']}/{target_info['majority_class']})
- Imbalance: {target_info['imbalance_ratio']:.1f}:1

BASELINE: {baseline_acc:.4f} (always predict majority class)

LOGISTIC REGRESSION:
- Test Accuracy: {lr_scores['test_acc']:.4f}
- Test F1: {lr_scores['f1']:.4f}
- Improvement vs baseline: +{(lr_scores['test_acc']-baseline_acc)*100:.2f}%

MODEL COMPARISON (defaults):
{results_df[['Model', 'Test Accuracy', 'Overfit Gap', 'vs Baseline']].to_string(index=False)}

TUNED MODELS:
{tuned_df[['Model', 'Test Accuracy', 'Test F1']].to_string(index=False)}

BEST MODEL: {best_model_name}

TOP FEATURES:
{chr(10).join(f"- {f['feature']}: {f['coefficient']:.4f}" for f in lr_scores['top_features'][:5])}
"""
    print(summary)
    save_output("final_summary.txt", summary)
    
    print("\n" + "=" * 80)
    print(" COMPLETE - All outputs saved to outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
