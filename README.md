# Bank Marketing Classifier Comparison

## Project Overview

This project compares the performance of four classification algorithms (K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines) on a bank marketing dataset from a Portuguese banking institution. The goal is to predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign data.

## Dataset

- **Source**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size**: 41,188 contacts from 17 marketing campaigns (May 2008 - November 2010)
- **Features**: 20 input variables (10 numeric, 10 categorical)
- **Target**: Binary classification - whether client subscribed to term deposit (yes/no)
- **Class Imbalance**: 88.73% 'no' vs 11.27% 'yes' (7.9:1 ratio)

## Key Findings

### Model Performance

| Model | Test Accuracy | Test F1 | Notes |
|-------|--------------|---------|-------|
| Logistic Regression | 0.9014 | 0.3388 | Good balance of speed and accuracy |
| KNN | 0.8940 | 0.3751 | Fast training, moderate performance |
| Decision Tree | 0.8412 | 0.3986 | Best F1 after tuning, but overfits with defaults |
| SVM | 0.9037 | 0.3711 | Highest accuracy, slow training |

**Baseline accuracy**: 88.74% (always predicting 'no')

### Top Predictive Features

1. **emp.var.rate** (negative): Lower employment variation rate increases subscription likelihood
2. **month_mar** (positive): March campaigns have higher conversion
3. **cons.price.idx** (positive): Higher consumer price index correlates with subscriptions
4. **contact_telephone** (negative): Cellular contact is more effective than telephone
5. **poutcome_success** (positive): Previous campaign success strongly predicts current success

## Recommendations for the Bank

1. **Target Selection**: Prioritize clients with successful previous campaign outcomes
2. **Optimal Timing**: Focus campaigns in March, September, October, and December
3. **Contact Method**: Use cellular phones instead of landline telephones
4. **Economic Awareness**: Launch campaigns when employment metrics are favorable
5. **Model Deployment**: Use tuned Decision Tree for best minority class detection (F1=0.40)

## Repository Structure

```
comparing-classifiers/
├── data/
│   ├── bank-additional-full.csv    # Full dataset (41,188 rows)
│   ├── bank-additional.csv         # 10% sample
│   └── bank-additional-names.txt   # Data description
├── outputs/                        # Analysis outputs
│   ├── problem_*.txt               # Problem-specific findings
│   ├── final_summary.txt           # Complete summary
│   └── *.csv                       # Model results
├── visualizations/                 # Generated plots
├── prompt_III.ipynb               # Main Jupyter notebook
├── solution.py                    # Complete solution script
├── explore_data.py                # Generic data exploration tool
├── data_profile.txt               # Data profile output
├── CRISP-DM-BANK.pdf             # Research paper
└── README.md                      # This file
```

## How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the solution script:
```bash
python solution.py
```

3. Or open the Jupyter notebook:
```bash
jupyter notebook prompt_III.ipynb
```

## Methodology

1. **Data Understanding**: Analyzed 17 campaigns with 41,188 contacts
2. **Data Preparation**: Dropped 'duration' (data leakage), encoded categoricals, scaled numerics
3. **Modeling**: Trained 4 classifiers with default and tuned hyperparameters
4. **Evaluation**: Used F1-score as primary metric due to class imbalance
5. **Interpretation**: Analyzed feature coefficients for business insights

## Next Steps

1. Deploy model with performance monitoring
2. A/B test predictions against current targeting
3. Explore ensemble methods (Random Forest, XGBoost)
4. Collect additional features (customer tenure, account balances)
5. Build cost-benefit threshold analysis

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

