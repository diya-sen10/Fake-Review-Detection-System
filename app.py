
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# 2. LOAD MODEL AND TF-IDF VECTORIZER
# ==================================================
@st.cache_resource
def load_model():
    folder = Path(__file__).resolve().parent

    model_path = folder / "fake_review_model.pkl"
    vectorizer_path = folder / "tfidf_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}"
        )

    trained_model = joblib.load(model_path)
    trained_tfidf = joblib.load(vectorizer_path)

    return trained_model, trained_tfidf


try:
    model, tfidf = load_model()
    model_error = None
except Exception as error:
    model, tfidf = None, None
    model_error = str(error)


# ==================================================
# 3. SESSION STATE
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "full_review" not in st.session_state:
    st.session_state.full_review = ""


# ==================================================
# 4. WEBSITE DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111c30;
}

h1, h2, h3 {
    color: #e2e8f0 !important;
}

p, label, [data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #344563;
    padding: 18px;
    border-radius: 14px;
}

.stTextArea textarea {
    background: #17243a;
    color: white;
    border: 1px solid #64748b;
    border-radius: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: bold;
}

.stButton > button p {
    color: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #344563;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================
st.sidebar.markdown("# 🛡️ ReviewGuard")
st.sidebar.caption("AI-powered review analysis")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Review Analyzer",
        "Analytics",
        "Settings",
        "Help"
    ]
)


st.sidebar.divider()

st.sidebar.markdown("### 🤖 Model Information")

st.sidebar.markdown("**Model Type**")
st.sidebar.info("Logistic Regression")

st.sidebar.markdown("**Review Categories**")
st.sidebar.write("🚩 Fake")
st.sidebar.write("✅ Genuine")

st.sidebar.markdown("**TF-IDF Features**")
st.sidebar.metric("Text Features", "35,147")

st.sidebar.caption(
    "These features help the model process review text."
)


# ==================================================
# 6. MAIN TITLE
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;color:#a78bfa !important;">
        🛡️ Fake Review Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Intelligent product review analysis using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

if model_error:
    st.error("The model could not be loaded.")
    st.code(model_error)
    st.info(
        "Keep app.py, fake_review_model.pkl, and "
        "tfidf_vectorizer.pkl in the same folder."
    )


# ==================================================
# 7. PREDICTION FUNCTION
# ==================================================
def analyze_review(review):
    if model is None or tfidf is None:
        raise RuntimeError(
            "Model files are not loaded. Check the error above."
        )

    if not review or not review.strip():
        raise ValueError("Please enter a review first.")

    # Convert the review into TF-IDF features
    features = tfidf.transform([review.strip()])

    # Predict the class
    prediction = str(model.predict(features)[0]).strip()

    # Get class probabilities when supported
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    # Convert the model label into a readable result
    label = prediction.lower()

    if label in ["fake", "fake review", "or"]:
        result = "Fake Review"

    elif label in ["genuine", "genuine review", "cg"]:
        result = "Genuine Review"

    else:
        result = prediction

    # Confidence of the predicted class
    confidence = None

    if probabilities is not None:
        classes = [
            str(value).strip()
            for value in model.classes_
        ]

        predicted_index = classes.index(prediction)

        confidence = float(
            probabilities[predicted_index] * 100
        )

    # Debug information appears in the VS Code terminal
    print("Input review:", review)
    print("Model classes:", model.classes_)
    print("Raw prediction:", prediction)
    print("Probabilities:", probabilities)

    return result, confidence


# ==================================================
# 8. SAVE AND DISPLAY PREDICTIONS
# ==================================================
def save_prediction(review, result, confidence):
    item = {
        "Review": review,
        "Prediction": result,
        "Confidence": (
            f"{confidence:.2f}%"
            if confidence is not None
            else "Unavailable"
        ),
        "Time": datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )
    }

    st.session_state.history.insert(0, item)
    st.session_state.last_result = item


def show_prediction(result, confidence):
    if result == "Fake Review":
        st.error("🚩 Prediction: Fake Review")

    elif result == "Genuine Review":
        st.success("✅ Prediction: Genuine Review")

    else:
        st.info(f"Prediction: {result}")

    if confidence is not None:
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )
    else:
        st.info("Confidence is not available.")

    st.caption(
        "A model prediction is an estimate, "
        "not proof that a review is fake."
    )


# ==================================================
# 9. DASHBOARD
# ==================================================
if page == "Dashboard":

    total = len(st.session_state.history)

    fake = sum(
        item["Prediction"] == "Fake Review"
        for item in st.session_state.history
    )

    genuine = sum(
        item["Prediction"] == "Genuine Review"
        for item in st.session_state.history
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📝 Reviews Analyzed", total)
    col2.metric("🚩 Fake Predictions", fake)
    col3.metric("✅ Genuine Predictions", genuine)

    fake_rate = fake / total * 100 if total else 0

    col4.metric("📊 Fake Rate", f"{fake_rate:.1f}%")

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:

        st.subheader("📈 Prediction Overview")

        if total > 0:
            chart_data = pd.DataFrame({
                "Category": ["Fake", "Genuine"],
                "Count": [fake, genuine]
            })

            st.bar_chart(
                chart_data.set_index("Category"),
                color="#8b5cf6"
            )

        else:
            st.info(
                "Analyze your first review to generate statistics."
            )

        st.subheader("🕘 Recent Reviews")

        if st.session_state.history:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.history[:5]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption(
                "Your recent predictions will appear here."
            )

    with right:

        st.subheader("⚡ Quick Review Analyzer")

        quick_review = st.text_area(
            "Enter a product review",
            placeholder="Type or paste a review...",
            height=180,
            key="dashboard_review"
        )

        if st.button(
            "🔍 Analyze Review",
            key="dashboard_analyze"
        ):

            if not quick_review.strip():
                st.warning("Please enter a review.")

            else:
                try:
                    result, confidence = analyze_review(
                        quick_review
                    )

                    save_prediction(
                        quick_review,
                        result,
                        confidence
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Prediction error: {error}"
                    )

        # Display the latest result
        if st.session_state.last_result:

            latest = st.session_state.last_result

            st.subheader("Latest Prediction")

            latest_confidence = None

            if latest["Confidence"] != "Unavailable":
                latest_confidence = float(
                    latest["Confidence"].replace("%", "")
                )

            show_prediction(
                latest["Prediction"],
                latest_confidence
            )


# ==================================================
# 10. FULL REVIEW ANALYZER
# ==================================================
elif page == "Review Analyzer":

    st.subheader("✍️ Analyze a Product Review")

    st.write(
        "Paste a review below or choose a sample review."
    )

    sample_reviews = {
        "Fake-style": [
            "Best product ever!",
            "Amazing quality!",
            "Love it so much!",
            "Must buy this!",
            "Super nice product!",
            "Totally worth it!",
            "Best thing ever!",
            "Very happy with it!",
            "Just buy it!",
            "Five stars!"
        ],

        "Genuine-style": [
            "Works as expected.",
            "Good for the price.",
            "Size is too small.",
            "It came late.",
            "Looks like the photo.",
            "Works fine for me.",
            "Quality is okay.",
            "The color is nice.",
            "Not worth the price.",
            "It is easy to use."
        ]
    }

    review_type = st.selectbox(
        "Choose sample category",
        ["Choose a category"] + list(sample_reviews.keys())
    )

    if review_type != "Choose a category":

        selected_review = st.selectbox(
            "Choose a sample review",
            sample_reviews[review_type],
            key="sample_review_choice"
        )

        # Callback runs before Streamlit redraws the page
        def use_selected_sample():
            st.session_state.full_review = (
                st.session_state.sample_review_choice
            )

        st.button(
            "Use This Review",
            on_click=use_selected_sample
        )

    review = st.text_area(
        "Product review",
        placeholder="Type, paste, or select a sample review...",
        height=220,
        key="full_review"
    )

    if st.button(
        "🔍 Analyze Review",
        key="full_analyze"
    ):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            try:
                result, confidence = analyze_review(review)

                save_prediction(
                    review,
                    result,
                    confidence
                )

                show_prediction(
                    result,
                    confidence
                )

            except Exception as error:
                st.error(
                    f"Prediction error: {error}"
                )


# ==================================================
# 11. ANALYTICS
# ==================================================
elif page == "Analytics":

    st.subheader("📊 Review Analytics")

    if not st.session_state.history:

        st.info(
            "Analyze reviews first to see analytics."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        counts = history_df["Prediction"].value_counts()

        st.bar_chart(
            counts,
            color="#8b5cf6"
        )

        st.subheader("All Analyzed Reviews")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Export History as CSV",
            data=csv,
            file_name="review_history.csv",
            mime="text/csv"
        )


# ==================================================
# 12. SETTINGS
# ==================================================
elif page == "Settings":

    st.subheader("⚙️ Settings")

    st.write("Model: Saved Machine Learning classifier")
    st.write("Text processing: TF-IDF vectorizer")

    if model is not None and tfidf is not None:

        st.write(
            "Model classes:",
            list(model.classes_)
        )

        st.write(
            "Expected model features:",
            model.n_features_in_
        )

        st.write(
            "Vectorizer features:",
            len(tfidf.get_feature_names_out())
        )

    else:
        st.warning(
            "Model details are unavailable because "
            "the model files did not load."
        )

    if st.button("🗑️ Clear Review History"):

        st.session_state.history = []
        st.session_state.last_result = None

        st.success("Review history cleared.")


# ==================================================
# 13. HELP
# ==================================================
elif page == "Help":

    st.subheader("❓ How to Use This System")

    st.markdown("""
    1. Open **Review Analyzer**.
    2. Paste a product review.
    3. Click **Analyze Review**.
    4. Read the prediction and confidence score.
    5. Open **Dashboard** or **Analytics** to see results.

    **Fake Review:** The model predicts that the review
    belongs to the fake-review class.

    **Genuine Review:** The model predicts that the review
    belongs to the genuine-review class.

    **Important:** Confidence is not a guarantee of correctness.
    """)


# ==================================================
# 14. FOOTER
# ==================================================
st.divider()

st.caption(
    "🛡️ Fake Review Detection System | "
    "Python • Streamlit • Machine Learning • TF-IDF"
)