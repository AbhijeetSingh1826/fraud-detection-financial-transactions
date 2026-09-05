import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")

st.title("🔍 Real-Time Fraud Detection Dashboard")
st.markdown("Monitoring transaction stream for fraudulent activity")

# --- Load results from the stream simulation ---
@st.cache_data
def load_data():
    return pd.read_csv("../data/stream_results.csv")

results_df = load_data()

# --- Top-level metrics ---
col1, col2, col3, col4 = st.columns(4)

total_txns = len(results_df)
flagged = results_df['predicted_fraud'].sum()
actual_fraud = results_df['actual_fraud'].sum()
accuracy = (results_df['correct'].sum() / total_txns) * 100

col1.metric("Total Transactions", total_txns)
col2.metric("Flagged as Fraud", int(flagged))
col3.metric("Actual Fraud Cases", int(actual_fraud))
col4.metric("Model Accuracy on Stream", f"{accuracy:.1f}%")

st.divider()

# --- Risk level breakdown ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Risk Level Distribution")
    risk_counts = results_df['risk_level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                 color=risk_counts.index,
                 color_discrete_map={'HIGH': '#ff4b4b', 'MEDIUM': '#ffa94d', 'LOW': '#51cf66'})
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Fraud Probability Distribution")
    fig2 = px.histogram(results_df, x='fraud_probability', nbins=20,
                         color='actual_fraud',
                         labels={'actual_fraud': 'Actual Fraud'})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Live-style transaction feed table ---
st.subheader("Transaction Feed")

def highlight_fraud(row):
    if row['predicted_fraud'] == 1:
        return ['background-color: #ffe3e3'] * len(row)
    return [''] * len(row)

st.dataframe(
    results_df.style.apply(highlight_fraud, axis=1),
    use_container_width=True,
    height=400
)

st.divider()

# --- Try it yourself: manual transaction check ---
st.subheader("Test a Transaction Manually")
st.markdown("Pick a transaction from the test set and see the model's live prediction")

idx = st.slider("Select a transaction row from the stream", 0, len(results_df)-1, 0)
selected = results_df.iloc[idx]

st.write(f"**Prediction:** {'🚨 FRAUD' if selected['predicted_fraud'] == 1 else '✅ Legit'}")
st.write(f"**Fraud Probability:** {selected['fraud_probability']:.3f}")
st.write(f"**Risk Level:** {selected['risk_level']}")
st.write(f"**Actual Label:** {'Fraud' if selected['actual_fraud'] == 1 else 'Legit'}")