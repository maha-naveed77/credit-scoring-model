import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a237e; }
    .sub-header  { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .step-box    { background: #f0f4ff; border-left: 4px solid #3f51b5;
                   padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
    .result-good { background: #e8f5e9; border: 2px solid #4caf50;
                   padding: 20px; border-radius: 10px; text-align: center; }
    .result-bad  { background: #ffebee; border: 2px solid #f44336;
                   padding: 20px; border-radius: 10px; text-align: center; }
    .metric-row  { display: flex; gap: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load Saved Artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf            = joblib.load('model/random_forest.pkl')
    scaler        = joblib.load('model/scaler.pkl')
    encoders      = joblib.load('model/encoders.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    defaults      = joblib.load('model/defaults.pkl')
    return rf, scaler, encoders, feature_names, defaults

try:
    rf, scaler, encoders, feature_names, defaults = load_artifacts()
    model_ready = True
except FileNotFoundError:
    model_ready = False


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">💳 Credit Scoring Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predicts whether a bank applicant is a <b>Good</b> or <b>Bad</b> '
    'credit risk using a Random Forest classifier trained on 1000 real German bank applicants.</div>',
    unsafe_allow_html=True
)

if not model_ready:
    st.error("⚠️ Model files not found. Please run `python train.py` first in your terminal.")
    st.code("python train.py", language="bash")
    st.stop()

st.markdown("---")


# ── How It Works (collapsible) ─────────────────────────────────────────────────
with st.expander("ℹ️ How does this work? — Full ML Pipeline Explained"):
    st.markdown("""
    ### The 6-Step Pipeline

    | Step | What Happens | Why |
    |------|-------------|-----|
    | 1. Raw Input | You fill in applicant details | Give the model something to work with |
    | 2. Feature Engineering | App computes `credit_per_month` and `debt_to_duration` | These turned out to be the #1 and #2 most predictive features |
    | 3. Encoding | Text categories converted to numbers | ML models only understand numbers |
    | 4. Scaling | All numbers standardized (mean=0, std=1) | Prevents large-scale features from dominating |
    | 5. Random Forest | 100 decision trees each vote on the outcome | Majority vote gives final prediction |
    | 6. Output | Good ✅ or Bad ❌ with confidence % | Actionable result for the bank |

    ### Why Random Forest over Logistic Regression?
    | Model | Accuracy | ROC-AUC | Bad Credit F1 |
    |-------|----------|---------|---------------|
    | Logistic Regression | 69% | 0.73 | 0.42 |
    | **Random Forest** | **77%** | **0.79** | **0.52** |

    Random Forest wins on all metrics because it captures non-linear relationships between features
    that Logistic Regression (which draws a straight decision boundary) cannot.

    ### Why is Bad Credit harder to predict?
    The dataset has **700 good vs 300 bad** borrowers. The model sees far more "good" examples,
    so it naturally leans toward predicting "good." This is a real-world class imbalance problem —
    in production, banks use techniques like SMOTE (oversampling) or cost-sensitive learning to fix this.
    """)

st.markdown("---")


# ── Sidebar — Applicant Input Form ─────────────────────────────────────────────
st.sidebar.markdown("## 📋 Applicant Details")
st.sidebar.markdown("Fill in the information below and click **Predict**.")
st.sidebar.markdown("---")

st.sidebar.markdown("### 👤 Personal Info")
age           = st.sidebar.slider("Age", 18, 75, 35, help="Applicant's age in years")

st.sidebar.markdown("### 💰 Loan Details")
credit_amount = st.sidebar.number_input(
    "Loan Amount (DM)", min_value=500, max_value=20000, value=3000, step=100,
    help="Total loan amount in Deutsche Marks"
)
duration      = st.sidebar.slider(
    "Loan Duration (months)", 6, 72, 24,
    help="Number of months to repay the loan"
)

st.sidebar.markdown("### 🏦 Financial Status")
checking_opts = list(encoders['checking_status'].classes_)
checking      = st.sidebar.selectbox(
    "Checking Account Status", checking_opts,
    help="<0 = overdrawn | 0<=X<200 = low | >=200 = healthy | no checking = no account"
)

savings_opts  = list(encoders['savings_status'].classes_)
savings       = st.sidebar.selectbox(
    "Savings Account Status", savings_opts,
    help="How much is in the savings account"
)

st.sidebar.markdown("### 📜 Background")
ch_opts       = list(encoders['credit_history'].classes_)
credit_hist   = st.sidebar.selectbox(
    "Credit History", ch_opts,
    help="Past credit repayment behaviour"
)

emp_opts      = list(encoders['employment'].classes_)
employment    = st.sidebar.selectbox(
    "Employment Duration", emp_opts,
    help="How long applicant has been in current job"
)

purpose_opts  = list(encoders['purpose'].classes_)
purpose       = st.sidebar.selectbox(
    "Loan Purpose", purpose_opts,
    help="What the loan will be used for"
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(
    "🔍 Predict Credit Risk",
    use_container_width=True,
    type="primary"
)


# ── Main Dashboard — Always Visible ────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

# Feature Importance Chart
with col_left:
    st.subheader("📊 Feature Importance")
    st.caption("Which factors the Random Forest relies on most — across all 1000 applicants")

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = 12

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors  = ['#1a237e' if i < 3 else '#5c6bc0' if i < 7 else '#9fa8da'
               for i in range(top_n)]
    ax.bar(range(top_n), importances[indices[:top_n]], color=colors)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] for i in indices[:top_n]],
        rotation=40, ha='right', fontsize=8
    )
    ax.set_ylabel("Importance Score")
    ax.set_title("Top 12 Features by Importance", fontsize=11, fontweight='bold')

    legend_handles = [
        mpatches.Patch(color='#1a237e', label='Top 3 (most critical)'),
        mpatches.Patch(color='#5c6bc0', label='Ranks 4–7'),
        mpatches.Patch(color='#9fa8da', label='Ranks 8–12'),
    ]
    ax.legend(handles=legend_handles, fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        "💡 **credit_per_month** and **debt_to_duration** are the top 2 features — "
        "both were engineered from raw inputs, showing that feature engineering "
        "had a bigger impact than any original column alone."
    )

# Model Metrics
with col_right:
    st.subheader("📈 Model Performance")
    st.caption("Evaluated on 200 applicants the model had never seen during training")

    r1c1, r1c2 = st.columns(2)
    r1c1.metric("Random Forest Accuracy", "77%",  delta="vs 69% LR")
    r1c2.metric("ROC-AUC Score",          "0.79", delta="vs 0.73 LR")

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("Good Credit F1",  "0.84", help="How well it predicts reliable borrowers")
    r2c2.metric("Bad Credit F1",   "0.52", help="Harder — fewer training examples")

    st.markdown("---")
    st.markdown("#### 🔢 Confusion Matrix Breakdown")
    st.markdown("""
    From 200 test applicants:

    |  | Predicted Good | Predicted Bad |
    |--|---|---|
    | **Actually Good** | ✅ 128 correctly approved | ❌ 12 wrongly rejected |
    | **Actually Bad**  | ⚠️ 35 wrongly approved | ✅ 25 correctly flagged |

    The biggest risk for a bank: the **35 bad borrowers predicted as good** —
    these are loans that would likely default. Improving this requires either
    more balanced training data or adjusting the decision threshold.
    """)

st.markdown("---")


# ── Prediction Section ─────────────────────────────────────────────────────────
if predict_btn:
    st.subheader("🔍 Live Prediction — Step by Step")

    # ── Step 1: Show Input ──────────────────────────────────────────
    with st.expander("📥  Step 1 — Raw Input Captured", expanded=True):
        input_table = {
            "Field":  ["Age", "Loan Amount", "Duration",
                       "Checking Account", "Savings Account",
                       "Credit History", "Employment", "Purpose"],
            "Value":  [f"{age} years", f"DM {credit_amount:,}", f"{duration} months",
                       checking, savings, credit_hist, employment, purpose],
            "Role":   ["Personal context", "Size of loan requested", "Repayment timeline",
                       "Current liquidity", "Financial cushion",
                       "Repayment track record", "Job stability", "Risk category of loan"]
        }
        st.dataframe(pd.DataFrame(input_table), use_container_width=True, hide_index=True)

    # ── Step 2: Feature Engineering ────────────────────────────────
    credit_per_month = credit_amount / duration
    debt_to_duration = credit_amount / (age + 1)

    with st.expander("🔧  Step 2 — Feature Engineering", expanded=True):
        fe_col1, fe_col2 = st.columns(2)
        with fe_col1:
            st.markdown(f"""
            <div class="step-box">
            <b>credit_per_month</b><br>
            = {credit_amount:,} ÷ {duration}<br>
            = <b style="font-size:1.2rem">{credit_per_month:.2f} DM/month</b><br><br>
            <small>Monthly loan burden. The model found this is the single strongest
            predictor of credit risk — a very high monthly repayment relative to other
            applicants is a major red flag.</small>
            </div>
            """, unsafe_allow_html=True)
        with fe_col2:
            st.markdown(f"""
            <div class="step-box">
            <b>debt_to_duration</b><br>
            = {credit_amount:,} ÷ ({age} + 1)<br>
            = <b style="font-size:1.2rem">{debt_to_duration:.2f}</b><br><br>
            <small>Loan size relative to life stage. A young applicant with a large
            loan has fewer years of earning history to back it up — this ratio
            captures that context.</small>
            </div>
            """, unsafe_allow_html=True)

    # ── Step 3: Encode + Scale ──────────────────────────────────────
    with st.expander("🔢  Step 3 — Encoding & Scaling", expanded=True):
        enc_col1, enc_col2 = st.columns(2)
        with enc_col1:
            enc_checking = int(encoders['checking_status'].transform([checking])[0])
            enc_savings  = int(encoders['savings_status'].transform([savings])[0])
            enc_ch       = int(encoders['credit_history'].transform([credit_hist])[0])
            enc_emp      = int(encoders['employment'].transform([employment])[0])
            enc_pur      = int(encoders['purpose'].transform([purpose])[0])

            enc_table = {
                "Column":   ["checking_status", "savings_status", "credit_history",
                             "employment", "purpose"],
                "Original": [checking, savings, credit_hist, employment, purpose],
                "Encoded":  [enc_checking, enc_savings, enc_ch, enc_emp, enc_pur]
            }
            st.caption("Label Encoding — text → number:")
            st.dataframe(pd.DataFrame(enc_table), use_container_width=True, hide_index=True)
        with enc_col2:
            st.caption("StandardScaler — why we scale:")
            st.markdown("""
            `credit_amount` can be 500–20,000.  
            `age` is 18–75.  
            `credit_per_month` is 20–1,000.  

            Without scaling, the model would give more weight to large-scale features
            just because their numbers are bigger — not because they are more important.
            Scaling puts everything on the same level.
            """)

    # ── Step 4: Build Feature Vector & Predict ──────────────────────
    input_data = defaults.copy()
    input_data['age']             = float(age)
    input_data['credit_amount']   = float(credit_amount)
    input_data['duration']        = float(duration)
    input_data['checking_status'] = float(enc_checking)
    input_data['savings_status']  = float(enc_savings)
    input_data['credit_history']  = float(enc_ch)
    input_data['employment']      = float(enc_emp)
    input_data['purpose']         = float(enc_pur)
    input_data['credit_per_month'] = credit_per_month
    input_data['debt_to_duration'] = debt_to_duration

    input_array  = np.array([[input_data[f] for f in feature_names]])
    input_scaled = scaler.transform(input_array)

    prediction   = rf.predict(input_scaled)[0]
    probability  = rf.predict_proba(input_scaled)[0]
    good_prob    = probability[1]
    bad_prob     = probability[0]

    with st.expander("🌲  Step 4 — Random Forest Voting", expanded=True):
        good_votes = int(round(good_prob * 100))
        bad_votes  = 100 - good_votes
        st.markdown(f"""
        The 100 decision trees each independently analyzed this applicant's profile and voted:

        | Vote | Count | Interpretation |
        |------|-------|----------------|
        | ✅ Good Credit | **{good_votes} trees** | Patterns match a reliable borrower |
        | ❌ Bad Credit  | **{bad_votes} trees** | Patterns suggest repayment risk |

        **Majority wins → Final decision: {"✅ Good Credit" if prediction == 1 else "❌ Bad Credit"}**
        """)

    # ── Step 5: Final Result ────────────────────────────────────────
    st.subheader("📋 Final Prediction")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-good">
            <h2>✅ Good Credit Risk</h2>
            <h3>Confidence: {good_prob:.1%}</h3>
            <p>This applicant is predicted to be a <b>reliable borrower</b>.<br>
            The bank can likely approve this loan with reasonable confidence.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-bad">
            <h2>❌ Bad Credit Risk</h2>
            <h3>Confidence: {bad_prob:.1%}</h3>
            <p>This applicant is predicted to be a <b>high-risk borrower</b>.<br>
            The bank should review this application carefully before approving.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability visualization
    prob_col1, prob_col2 = st.columns([2, 1])
    with prob_col1:
        fig2, ax2 = plt.subplots(figsize=(7, 1.2))
        ax2.barh([''], [good_prob], color='#4caf50', label=f'Good  {good_prob:.1%}')
        ax2.barh([''], [bad_prob], left=[good_prob], color='#f44336', label=f'Bad  {bad_prob:.1%}')
        ax2.axvline(x=0.5, color='#333', linestyle='--', linewidth=1.5, label='Decision boundary (50%)')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        ax2.set_title('Prediction Confidence Breakdown', fontweight='bold')
        ax2.legend(loc='lower right', fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with prob_col2:
        st.metric("✅ Good Credit", f"{good_prob:.1%}")
        st.metric("❌ Bad Credit",  f"{bad_prob:.1%}")

    # ── Key factors in this prediction ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔑 Key Factors in This Prediction")

    # Compare applicant's top features vs dataset medians
    key_features = ['credit_per_month', 'debt_to_duration', 'credit_amount', 'duration', 'age']
    median_vals  = {f: defaults[f] for f in key_features}
    applic_vals  = {
        'credit_per_month': credit_per_month,
        'debt_to_duration': debt_to_duration,
        'credit_amount':    float(credit_amount),
        'duration':         float(duration),
        'age':              float(age)
    }
    labels       = ['credit_per_month', 'debt_to_duration', 'credit_amount (÷100)', 'duration', 'age']
    applicant_v  = [applic_vals[f] if 'amount' not in f else applic_vals[f]/100 for f in key_features]
    median_v     = [median_vals[f]  if 'amount' not in f else median_vals[f]/100  for f in key_features]

    fig3, ax3 = plt.subplots(figsize=(7, 3))
    x     = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, applicant_v, width, label='This Applicant', color='#3f51b5', alpha=0.85)
    ax3.bar(x + width/2, median_v,    width, label='Dataset Median',  color='#90a4ae', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax3.set_title('Applicant vs. Median Borrower — Top Features', fontweight='bold')
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.caption(
        "Blue = this applicant's values. Grey = typical (median) borrower in the dataset. "
        "Values above median on debt features = higher risk. "
        "credit_amount divided by 100 for readability."
    )

else:
    # Placeholder when no prediction yet
    st.info("👈  Fill in the applicant details in the **sidebar** and click **Predict Credit Risk** to see the full pipeline in action.")

    st.markdown("#### 💡 What you'll see after clicking Predict:")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **Steps 1–2**
    - Raw input table
    - Feature engineering with live calculations
    """)
    c2.markdown("""
    **Steps 3–4**
    - Label encoding (text → numbers)
    - Random Forest tree voting breakdown
    """)
    c3.markdown("""
    **Step 5: Result**
    - Good ✅ or Bad ❌ with confidence %
    - Probability bar chart
    - Applicant vs. median comparison chart
    """)
