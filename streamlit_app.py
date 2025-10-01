import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI Model Comparison: Astra vs Qwen")
st.markdown("### Ground Truth: Human-labeled Database (DB)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
st.sidebar.info(f"Total Records: {len(df)}")

# Filters Section
st.sidebar.header("🔍 Data Filters")

# Get unique values for each DB field
specialty_options = ['All'] + sorted(df['db_specialty'].unique().tolist())
equipment_options = ['All'] + sorted(df['db_has_equipments'].unique().tolist())
coman_options = ['All'] + sorted(df['db_is_coman'].unique().tolist())
fb_options = ['All'] + sorted(df['db_is_food_beverage'].unique().tolist())

# Filter dropdowns
filter_specialty = st.sidebar.selectbox("DB Specialty", specialty_options)
filter_equipment = st.sidebar.selectbox("DB Has Equipment", equipment_options)
filter_coman = st.sidebar.selectbox("DB Is Co-Manufacturing", coman_options)
filter_fb = st.sidebar.selectbox("DB Is Food & Beverage", fb_options)

# Apply filters
df_filtered = df.copy()
if filter_specialty != 'All':
    df_filtered = df_filtered[df_filtered['db_specialty'] == filter_specialty]
if filter_equipment != 'All':
    df_filtered = df_filtered[df_filtered['db_has_equipments'] == filter_equipment]
if filter_coman != 'All':
    df_filtered = df_filtered[df_filtered['db_is_coman'] == filter_coman]
if filter_fb != 'All':
    df_filtered = df_filtered[df_filtered['db_is_food_beverage'] == filter_fb]

st.sidebar.info(f"Filtered Records: {len(df_filtered)}")

# Show active filters count
active_filters = sum([
    filter_specialty != 'All',
    filter_equipment != 'All',
    filter_coman != 'All',
    filter_fb != 'All'
])
if active_filters > 0:
    st.sidebar.warning(f"⚠️ {active_filters} filter(s) active")

st.sidebar.markdown("---")
show_confusion = st.sidebar.checkbox("Show Confusion Matrices", value=True)
show_errors = st.sidebar.checkbox("Show Error Analysis", value=False)
show_disagreements = st.sidebar.checkbox("Show Model Disagreements", value=False)

# Helper functions
def calculate_metrics(df, db_col, model_col, is_specialty=False):
    """Calculate accuracy, precision, and confusion matrix metrics"""
    if is_specialty:
        correct = (df[db_col] == df[model_col]).sum()
        total = len(df)
        accuracy = correct / total * 100
        errors = total - correct

        return {
            'accuracy': accuracy,
            'precision': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors
        }
    else:
        db_vals = df[db_col].astype(str).str.lower()
        model_vals = df[model_col].astype(str).str.lower()

        db_bool = db_vals.isin(['true', 'yes', '1'])
        model_bool = model_vals.isin(['true', 'yes', '1'])

        tp = ((db_bool == True) & (model_bool == True)).sum()
        tn = ((db_bool == False) & (model_bool == False)).sum()
        fp = ((db_bool == False) & (model_bool == True)).sum()
        fn = ((db_bool == True) & (model_bool == False)).sum()

        total = len(df)
        accuracy = (tp + tn) / total * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total
        }

# Categories
categories = {
    'Equipment': ('db_has_equipments', 'astra_has_equipments', 'qwen_has_equipments'),
    'Co-Manufacturing': ('db_is_coman', 'astra_is_coman', 'qwen_is_coman'),
    'Food & Beverage': ('db_is_food_beverage', 'astra_is_food_beverage', 'qwen_is_food_beverage'),
    'Specialty': ('db_specialty', 'astra_specialty', 'qwen_specialty')
}

# Calculate all metrics (using filtered data)
metrics_results = {}
for category, (db_col, astra_col, qwen_col) in categories.items():
    is_specialty = (category == 'Specialty')
    metrics_results[category] = {
        'astra': calculate_metrics(df_filtered, db_col, astra_col, is_specialty),
        'qwen': calculate_metrics(df_filtered, db_col, qwen_col, is_specialty)
    }

# Overall Summary Cards
st.markdown("---")
st.header("📈 Overall Performance")

# Calculate overall accuracy
astra_total_correct = sum(
    m['astra']['tp'] + m['astra']['tn'] if 'tp' in m['astra'] else m['astra']['correct']
    for m in metrics_results.values()
)
qwen_total_correct = sum(
    m['qwen']['tp'] + m['qwen']['tn'] if 'tp' in m['qwen'] else m['qwen']['correct']
    for m in metrics_results.values()
)
total_predictions = len(df_filtered) * 4

astra_overall = (astra_total_correct / total_predictions) * 100
qwen_overall = (qwen_total_correct / total_predictions) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🟦 ASTRA Overall Accuracy",
        value=f"{astra_overall:.2f}%",
        delta=f"{astra_overall - qwen_overall:+.2f}% vs QWEN"
    )

with col2:
    st.metric(
        label="🟥 QWEN Overall Accuracy",
        value=f"{qwen_overall:.2f}%",
        delta=f"{qwen_overall - astra_overall:+.2f}% vs ASTRA"
    )

with col3:
    winner = "QWEN" if qwen_overall > astra_overall else "ASTRA" if astra_overall > qwen_overall else "TIE"
    st.metric(
        label="🏆 Overall Winner",
        value=winner,
        delta=f"Margin: {abs(astra_overall - qwen_overall):.2f}%"
    )

# Accuracy Comparison Chart
st.markdown("---")
st.header("📊 Category-wise Performance Comparison")

cat_names = list(categories.keys())
astra_accuracies = [metrics_results[cat]['astra']['accuracy'] for cat in cat_names]
qwen_accuracies = [metrics_results[cat]['qwen']['accuracy'] for cat in cat_names]

fig = go.Figure()
fig.add_trace(go.Bar(
    name='ASTRA',
    x=cat_names,
    y=astra_accuracies,
    marker_color='#3498db',
    text=[f"{acc:.2f}%" for acc in astra_accuracies],
    textposition='outside'
))
fig.add_trace(go.Bar(
    name='QWEN',
    x=cat_names,
    y=qwen_accuracies,
    marker_color='#e74c3c',
    text=[f"{acc:.2f}%" for acc in qwen_accuracies],
    textposition='outside'
))

fig.update_layout(
    title="Accuracy Comparison Across Categories",
    xaxis_title="Category",
    yaxis_title="Accuracy (%)",
    barmode='group',
    height=500,
    yaxis=dict(range=[0, 105])
)

st.plotly_chart(fig, use_container_width=True)

# Precision Comparison (for boolean categories)
bool_cats = ['Equipment', 'Co-Manufacturing', 'Food & Beverage']
astra_precisions = [metrics_results[cat]['astra']['precision'] for cat in bool_cats]
qwen_precisions = [metrics_results[cat]['qwen']['precision'] for cat in bool_cats]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name='ASTRA',
    x=bool_cats,
    y=astra_precisions,
    marker_color='#2ecc71',
    text=[f"{prec:.2f}%" for prec in astra_precisions],
    textposition='outside'
))
fig2.add_trace(go.Bar(
    name='QWEN',
    x=bool_cats,
    y=qwen_precisions,
    marker_color='#f39c12',
    text=[f"{prec:.2f}%" for prec in qwen_precisions],
    textposition='outside'
))

fig2.update_layout(
    title="Precision Comparison (Boolean Categories)",
    xaxis_title="Category",
    yaxis_title="Precision (%)",
    barmode='group',
    height=500,
    yaxis=dict(range=[0, 105])
)

st.plotly_chart(fig2, use_container_width=True)

# Detailed Category Analysis
st.markdown("---")
st.header("🔍 Detailed Category Analysis")

selected_category = st.selectbox("Select Category for Detailed Analysis", cat_names)

col1, col2 = st.columns(2)

astra_m = metrics_results[selected_category]['astra']
qwen_m = metrics_results[selected_category]['qwen']

with col1:
    st.subheader("🟦 ASTRA Performance")
    st.metric("Accuracy", f"{astra_m['accuracy']:.2f}%")
    st.metric("Precision", f"{astra_m['precision']:.2f}%")

    if 'tp' in astra_m:
        st.write(f"**True Positives:** {astra_m['tp']}")
        st.write(f"**True Negatives:** {astra_m['tn']}")
        st.write(f"**False Positives:** {astra_m['fp']} ⚠️")
        st.write(f"**False Negatives:** {astra_m['fn']} ⚠️")

with col2:
    st.subheader("🟥 QWEN Performance")
    st.metric("Accuracy", f"{qwen_m['accuracy']:.2f}%")
    st.metric("Precision", f"{qwen_m['precision']:.2f}%")

    if 'tp' in qwen_m:
        st.write(f"**True Positives:** {qwen_m['tp']}")
        st.write(f"**True Negatives:** {qwen_m['tn']}")
        st.write(f"**False Positives:** {qwen_m['fp']} ⚠️")
        st.write(f"**False Negatives:** {qwen_m['fn']} ⚠️")

# Confusion Matrices
if show_confusion and 'tp' in astra_m:
    st.markdown("---")
    st.subheader("🎯 Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**ASTRA Confusion Matrix**")
        cm_astra = np.array([[astra_m['tp'], astra_m['fn']],
                             [astra_m['fp'], astra_m['tn']]])

        fig_cm1 = px.imshow(cm_astra,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Positive', 'Negative'],
                           y=['Positive', 'Negative'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm1.update_layout(height=400)
        st.plotly_chart(fig_cm1, use_container_width=True)

    with col2:
        st.write("**QWEN Confusion Matrix**")
        cm_qwen = np.array([[qwen_m['tp'], qwen_m['fn']],
                           [qwen_m['fp'], qwen_m['tn']]])

        fig_cm2 = px.imshow(cm_qwen,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Positive', 'Negative'],
                           y=['Positive', 'Negative'],
                           text_auto=True,
                           color_continuous_scale='Reds')
        fig_cm2.update_layout(height=400)
        st.plotly_chart(fig_cm2, use_container_width=True)

# False Positives and False Negatives
st.markdown("---")
st.header("⚠️ False Positives & False Negatives Analysis")

fp_data = []
fn_data = []

for cat in bool_cats:
    fp_data.append({
        'Category': cat,
        'ASTRA': metrics_results[cat]['astra']['fp'],
        'QWEN': metrics_results[cat]['qwen']['fp']
    })
    fn_data.append({
        'Category': cat,
        'ASTRA': metrics_results[cat]['astra']['fn'],
        'QWEN': metrics_results[cat]['qwen']['fn']
    })

fp_df = pd.DataFrame(fp_data)
fn_df = pd.DataFrame(fn_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("False Positives")
    fig_fp = go.Figure()
    fig_fp.add_trace(go.Bar(name='ASTRA', x=fp_df['Category'], y=fp_df['ASTRA'], marker_color='#e74c3c'))
    fig_fp.add_trace(go.Bar(name='QWEN', x=fp_df['Category'], y=fp_df['QWEN'], marker_color='#c0392b'))
    fig_fp.update_layout(barmode='group', height=400)
    st.plotly_chart(fig_fp, use_container_width=True)

with col2:
    st.subheader("False Negatives")
    fig_fn = go.Figure()
    fig_fn.add_trace(go.Bar(name='ASTRA', x=fn_df['Category'], y=fn_df['ASTRA'], marker_color='#f39c12'))
    fig_fn.add_trace(go.Bar(name='QWEN', x=fn_df['Category'], y=fn_df['QWEN'], marker_color='#d68910'))
    fig_fn.update_layout(barmode='group', height=400)
    st.plotly_chart(fig_fn, use_container_width=True)

# Model Disagreements
if show_disagreements:
    st.markdown("---")
    st.header("🔀 Model Disagreements")

    disagreement_category = st.selectbox("Select Category for Disagreement Analysis", cat_names)

    db_col, astra_col, qwen_col = categories[disagreement_category]

    if disagreement_category == 'Specialty':
        disagree_mask = df_filtered[astra_col] != df_filtered[qwen_col]
    else:
        disagree_mask = df_filtered[astra_col] != df_filtered[qwen_col]

    disagreements = df_filtered[disagree_mask]

    st.write(f"**Total Disagreements:** {len(disagreements)} out of {len(df_filtered)} ({len(disagreements)/len(df_filtered)*100:.2f}%)")

    if len(disagreements) > 0:
        st.dataframe(
            disagreements[['manufacturer_id', 'db_domain', db_col, astra_col, qwen_col]].head(20),
            use_container_width=True
        )

# Error Analysis
if show_errors:
    st.markdown("---")
    st.header("🐛 Error Analysis")

    error_category = st.selectbox("Select Category for Error Analysis", cat_names, key='error_cat')
    db_col, astra_col, qwen_col = categories[error_category]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ASTRA Errors")
        if error_category == 'Specialty':
            astra_errors = df_filtered[df_filtered[db_col] != df_filtered[astra_col]]
        else:
            astra_errors = df_filtered[df_filtered[db_col].astype(str).str.lower() != df_filtered[astra_col].astype(str).str.lower()]

        st.write(f"**Total Errors:** {len(astra_errors)}")
        if len(astra_errors) > 0:
            st.dataframe(
                astra_errors[['manufacturer_id', 'db_domain', db_col, astra_col]].head(10),
                use_container_width=True
            )

    with col2:
        st.subheader("QWEN Errors")
        if error_category == 'Specialty':
            qwen_errors = df_filtered[df_filtered[db_col] != df_filtered[qwen_col]]
        else:
            qwen_errors = df_filtered[df_filtered[db_col].astype(str).str.lower() != df_filtered[qwen_col].astype(str).str.lower()]

        st.write(f"**Total Errors:** {len(qwen_errors)}")
        if len(qwen_errors) > 0:
            st.dataframe(
                qwen_errors[['manufacturer_id', 'db_domain', db_col, qwen_col]].head(10),
                use_container_width=True
            )

# Summary Table
st.markdown("---")
st.header("📋 Summary Table")

summary_data = {
    'Category': cat_names + ['OVERALL'],
    'ASTRA Accuracy': [f"{metrics_results[cat]['astra']['accuracy']:.2f}%" for cat in cat_names] + [f"{astra_overall:.2f}%"],
    'QWEN Accuracy': [f"{metrics_results[cat]['qwen']['accuracy']:.2f}%" for cat in cat_names] + [f"{qwen_overall:.2f}%"],
    'Winner': [
        'ASTRA' if metrics_results[cat]['astra']['accuracy'] > metrics_results[cat]['qwen']['accuracy']
        else 'QWEN' if metrics_results[cat]['qwen']['accuracy'] > metrics_results[cat]['astra']['accuracy']
        else 'TIE'
        for cat in cat_names
    ] + ['ASTRA' if astra_overall > qwen_overall else 'QWEN' if qwen_overall > astra_overall else 'TIE']
}

summary_df = pd.DataFrame(summary_data)

# Color code the winner column
def highlight_winner(val):
    if val == 'ASTRA':
        return 'background-color: #3498db; color: white'
    elif val == 'QWEN':
        return 'background-color: #e74c3c; color: white'
    else:
        return 'background-color: #95a5a6; color: white'

styled_df = summary_df.style.map(highlight_winner, subset=['Winner'])
st.dataframe(styled_df, use_container_width=True)

# Key Insights
st.markdown("---")
st.header("💡 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟦 ASTRA Strengths")
    st.write("✅ Better at Equipment detection")
    st.write("✅ Better at Co-Manufacturing classification")
    st.write("✅ Higher precision in most boolean categories")

    st.subheader("🟦 ASTRA Weaknesses")
    st.write("❌ Struggles with Specialty classification (63.90%)")
    st.write("❌ Some false negatives in Food & Beverage")

with col2:
    st.subheader("🟥 QWEN Strengths")
    st.write("✅ Perfect Food & Beverage classification (100%)")
    st.write("✅ Significantly better Specialty classification (80.51%)")
    st.write("✅ Higher overall accuracy")

    st.subheader("🟥 QWEN Weaknesses")
    st.write("❌ Slightly more false negatives in Equipment")
    st.write("❌ Marginally lower precision in Co-Manufacturing")

st.markdown("---")
st.subheader("🎯 Recommendations")
st.info("""
1. **Ensemble Approach**: Combine both models for optimal results
2. **Use QWEN for**: Specialty and Food & Beverage classifications
3. **Use ASTRA for**: Equipment and Co-Manufacturing when precision is critical
4. **Improvement Focus**: Enhance ASTRA's Specialty classification capabilities
5. **Production Strategy**: Consider category-specific model routing based on strengths
""")

# Footer
st.markdown("---")
st.caption("Dashboard created for AI Model Comparison Analysis | Data Source: Human-labeled Ground Truth Database")