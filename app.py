import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Device Fault Detector", page_icon="⚡")

st.markdown("""
<style>
html, body, [class*="css"]          { font-size: 18px; }
h1                                  { font-size: 2.4rem !important; }
h2, h3                              { font-size: 1.5rem !important; }
div[data-testid="stMetricValue"]    { font-size: 2.2rem !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"]    { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("# Device Fault Detector")
st.caption("Upload your test CSV and the model will flag which devices are faulty.")


@st.cache_resource
def load_model():
    return joblib.load("lgb_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Couldn't load the model — make sure `lgb_model.pkl` is in the same folder. Error: {e}")
    st.stop()

FEATURES = [f"F{str(i).zfill(2)}" for i in range(1, 48)]


st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "What would you like to do?",
    ["Just get predictions", "Evaluate against ground truth"]
)

st.subheader("Upload your data")
test_file = st.file_uploader("Test CSV (needs ID + F01 to F47 columns)", type=["csv"])

labels_file = None
if mode == "Evaluate against ground truth":
    labels_file = st.file_uploader(
        "Ground truth CSV (needs ID + CLASS columns)",
        type=["csv"],
        key="labels"
    )

if not test_file:
    st.info("Upload a file above to get started.")
    st.stop()


data = pd.read_csv(test_file)

# Normalising the class column name in case it's "Class" instead of "CLASS"
if "Class" in data.columns:
    data.rename(columns={"Class": "CLASS"}, inplace=True)

available_features = [col for col in FEATURES if col in data.columns]
if len(available_features) < 47:
    st.warning(f"Heads up: only found {len(available_features)} out of 47 expected feature columns.")

X = data[available_features]

with st.spinner("Running predictions..."):
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)

results = pd.DataFrame({
    "ID":    data["ID"] if "ID" in data.columns else range(1, len(predictions) + 1),
    "CLASS": predictions,
})


total      = len(predictions)
normal     = int((predictions == 0).sum())
faulty     = int((predictions == 1).sum())
fault_rate = faulty / total * 100

st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total devices", total)
col2.metric("Normal",        normal)
col3.metric("Faulty",        faulty)
col4.metric("Fault rate",    f"{fault_rate:.1f}%")


# Prediction table
st.subheader("Predictions")

display = results.head(50).copy()
display["Status"]     = display["CLASS"].map({0: "Normal", 1: "Faulty"})
display["Confidence"] = probabilities.max(axis=1)[:50].round(4)

st.dataframe(display, use_container_width=True, hide_index=True)
if total > 50:
    st.caption(f"Showing 50 of {total:,} rows.")


# Figure out where the ground truth is coming from
ground_truth = None

if labels_file is not None:
    label_data = pd.read_csv(labels_file)
    if "Class" in label_data.columns:
        label_data.rename(columns={"Class": "CLASS"}, inplace=True)
    if "CLASS" in label_data.columns:
        ground_truth = label_data["CLASS"].values

elif mode == "Evaluate against ground truth" and "CLASS" in data.columns:
    ground_truth = data["CLASS"].values

if ground_truth is not None and len(ground_truth) == total:
    acc  = accuracy_score(ground_truth, predictions)
    f1   = f1_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions)
    rec  = recall_score(ground_truth, predictions)
    try:
        auc = roc_auc_score(ground_truth, probabilities[:, 1])
    except Exception:
        auc = None

    st.subheader("How did the model do?")

    st.markdown(f"""
    <div style="background:#e8f5e9; border-left:6px solid #2e7d32;
                border-radius:8px; padding:16px 24px; margin-bottom:20px;">
        <span style="font-size:1rem; color:#2e7d32; font-weight:600;">ACCURACY</span><br>
        <span style="font-size:3rem; font-weight:800; color:#1b5e20;">{acc * 100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # Supporting metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 Score",  f"{f1:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall",    f"{rec:.4f}")
    m4.metric("ROC-AUC",   f"{auc:.4f}" if auc is not None else "N/A")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        confusion_matrix(ground_truth, predictions),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Faulty"],
        yticklabels=["Normal", "Faulty"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close()

    # Full breakdown by class
    st.subheader("Classification Report")
    report = classification_report(
        ground_truth, predictions,
        target_names=["Normal", "Faulty"],
        output_dict=True
    )
    report_df = pd.DataFrame(report).T.drop("accuracy", errors="ignore").round(4)
    st.dataframe(report_df, use_container_width=True)


# Download buttons
st.subheader("Download results")

dl1, dl2 = st.columns(2)

with dl1:
    st.download_button(
        label="⬇ Predictions only (ID + CLASS)",
        data=results.to_csv(index=False).encode(),
        file_name="Final_Submission.csv",
        mime="text/csv",
        use_container_width=True
    )

with dl2:
    full_results = results.copy()
    full_results["Confidence_Normal"] = probabilities[:, 0].round(4)
    full_results["Confidence_Faulty"] = probabilities[:, 1].round(4)
    full_results["Status"] = full_results["CLASS"].map({0: "Normal", 1: "Faulty"})

    st.download_button(
        label="⬇ Full report (with confidence scores)",
        data=full_results.to_csv(index=False).encode(),
        file_name="Predictions_Full.csv",
        mime="text/csv",
        use_container_width=True
    )