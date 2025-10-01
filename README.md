# AI Model Comparison Analysis

Comprehensive comparison analysis of **Astra** and **Qwen** AI models against human-labeled ground truth data.

## 📊 Analysis Summary

**Total Records:** 277 manufacturers

### Overall Results
- **ASTRA Overall Accuracy:** 87.00%
- **QWEN Overall Accuracy:** 90.97%
- **Winner:** QWEN (by 3.97%)

### Category Performance

| Category | ASTRA Accuracy | QWEN Accuracy | Winner |
|----------|---------------|---------------|---------|
| Equipment | 97.83% | 96.03% | ASTRA |
| Co-Manufacturing | 88.09% | 87.36% | ASTRA |
| Food & Beverage | 98.19% | 100.00% | QWEN |
| Specialty | 63.90% | 80.51% | QWEN |

## 📁 Files Created

### 1. **model_comparison_analysis.ipynb**
Interactive Jupyter notebook with:
- Detailed metrics for each category
- Confusion matrices
- Visual comparisons
- Error analysis
- Model disagreement analysis

**To run:**
```bash
jupyter notebook model_comparison_analysis.ipynb
```

**Required packages:**
```bash
pip install pandas matplotlib seaborn scikit-learn numpy jupyter
```

### 2. **streamlit_app.py**
Interactive web dashboard with:
- Real-time metric visualization
- Category-wise drill-down
- False positives/negatives analysis
- Model disagreement viewer
- Interactive charts

**To run:**
```bash
streamlit run streamlit_app.py
```

**Required packages:**
```bash
pip install streamlit pandas plotly numpy
```

### 3. **model_comparison.py**
Python script for programmatic analysis:
- Command-line execution
- Reusable `ModelComparator` class
- JSON export capability
- Error and disagreement finder

**To run:**
```bash
python model_comparison.py
```

**Usage in code:**
```python
from model_comparison import ModelComparator

# Initialize
comparator = ModelComparator('your_data.csv')

# Run analysis
results = comparator.run_full_analysis()

# Export to JSON
comparator.export_results('results.json', results)

# Find disagreements
disagreements = comparator.find_disagreements('Equipment')

# Find model errors
errors = comparator.find_errors('Specialty', model='astra')
```

## 🔍 Key Insights

### ASTRA Strengths
✅ Better at Equipment detection (97.83% accuracy)
✅ Better at Co-Manufacturing classification (88.09% accuracy)
✅ Fewer false positives in most categories
✅ Higher precision overall

### ASTRA Weaknesses
❌ Poor Specialty classification (63.90% accuracy)
❌ Some false negatives in Food & Beverage category

### QWEN Strengths
✅ Perfect Food & Beverage classification (100% accuracy)
✅ Significantly better Specialty classification (80.51% accuracy)
✅ Higher overall accuracy (90.97%)
✅ More balanced performance across categories

### QWEN Weaknesses
❌ Slightly more false negatives in Equipment detection (11 vs 6)
❌ Marginally lower precision in Co-Manufacturing

## 📈 Metrics Explained

### Accuracy
Percentage of correct predictions out of all predictions.
- **Formula:** (TP + TN) / Total

### Precision
Percentage of correct positive predictions out of all positive predictions.
- **Formula:** TP / (TP + FP)

### False Positives (FP)
Model predicted positive, but ground truth is negative.
- **Impact:** Over-classification

### False Negatives (FN)
Model predicted negative, but ground truth is positive.
- **Impact:** Under-classification/missed cases

## 🎯 Recommendations

1. **Ensemble Approach**: Combine both models for optimal results
   - Use QWEN for Specialty and Food & Beverage
   - Use ASTRA for Equipment and Co-Manufacturing

2. **Production Strategy**: Implement category-specific routing
   ```python
   def classify(data, category):
       if category in ['Specialty', 'Food & Beverage']:
           return qwen_model.predict(data)
       else:
           return astra_model.predict(data)
   ```

3. **Improvement Focus**:
   - Enhance ASTRA's Specialty classification (biggest gap)
   - Reduce QWEN's false negatives in Equipment detection
   - Investigate the 100 specialty errors made by ASTRA

4. **Validation**: Focus human review on:
   - Cases where models disagree
   - Specialty classifications (lowest accuracy)
   - False positives in Co-Manufacturing

## 🛠️ Installation

Install all required packages:

```bash
# For Jupyter notebook
pip install pandas matplotlib seaborn scikit-learn numpy jupyter

# For Streamlit app
pip install streamlit plotly

# All-in-one
pip install pandas matplotlib seaborn scikit-learn numpy jupyter streamlit plotly
```

## 📞 Support

For questions or issues with the analysis tools, refer to the inline documentation in each file.

---

**Analysis Date:** 2025-09-30
**Data Source:** Copy of False positive comparison - no_prompt_combined.csv
**Ground Truth:** Human-labeled Database (DB)