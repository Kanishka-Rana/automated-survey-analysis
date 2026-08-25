import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Survey Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM DASHBOARD DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #172554;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    background: #eef6ff;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SENTIMENT ENGINE
# =========================================================

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    if pd.isna(text) or str(text).strip() == "":
        return "Neutral"

    score = sia.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="dashboard-title">📊 Survey Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Automated customer feedback analysis for faster business decisions'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR — DATA SOURCE
# =========================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### 📂 Choose Data Source")

data_source = st.sidebar.radio(
    "Select an option:",
    [
        "🟢 Use Demo Dataset",
        "🔵 Upload Your Own CSV"
    ]
)


# =========================================================
# DEMO DATASET / UPLOAD
# =========================================================

df = None


# ---------- DEMO DATASET ----------

if data_source == "🟢 Use Demo Dataset":

    demo_path = "automated_survey_analysis_dataset.csv"

    if os.path.exists(demo_path):

        try:

            df = pd.read_csv(demo_path)

            st.sidebar.success(
                "✅ Demo dataset loaded"
            )

            st.success(
                f"✅ Demo dataset loaded successfully — "
                f"{len(df)} survey responses"
            )

        except Exception as e:

            st.error(
                f"Unable to read demo dataset: {e}"
            )

            st.stop()

    else:

        st.error(
            "❌ Demo dataset not found. "
            "Make sure the DATA folder is inside your project folder."
        )

        st.stop()


# ---------- USER UPLOAD ----------

else:

    st.sidebar.markdown(
        "Upload a CSV containing customer survey responses."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Choose Survey CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "👈 Upload a survey CSV from the sidebar to begin analysis."
        )

        st.markdown("### 🚀 How the tool works")

        st.write("""
        **1.** Upload your survey CSV  
        **2.** The system reads the responses  
        **3.** Customer feedback is analyzed  
        **4.** Ratings and categories are summarized  
        **5.** Interactive charts are generated  
        **6.** Business insights are created automatically
        """)

        st.stop()

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"✅ Uploaded file loaded successfully — "
            f"{len(df)} survey responses"
        )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded file: {e}"
        )

        st.stop()


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("👀 View Survey Data"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# AUTOMATIC COLUMN DETECTION
# =========================================================

columns_lower = {
    str(col).lower(): col
    for col in df.columns
}


# ---------- FEEDBACK ----------

feedback_candidates = [
    "feedback",
    "review",
    "comment",
    "comments",
    "customer_feedback"
]

feedback_column = None

for col in feedback_candidates:

    if col in columns_lower:

        feedback_column = columns_lower[col]

        break


# ---------- RATING ----------

rating_candidates = [
    "rating",
    "overall_rating",
    "customer_rating"
]

rating_column = None

for col in rating_candidates:

    if col in columns_lower:

        rating_column = columns_lower[col]

        break


# ---------- CATEGORY ----------

category_candidates = [
    "category",
    "service",
    "department",
    "product_category"
]

category_column = None

for col in category_candidates:

    if col in columns_lower:

        category_column = columns_lower[col]

        break


# =========================================================
# COLUMN SELECTION
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Survey Configuration</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    if feedback_column is None:

        feedback_column = st.selectbox(
            "💬 Feedback Column",
            df.columns
        )

    else:

        st.info(
            f"💬 Feedback: **{feedback_column}**"
        )


with col2:

    if rating_column is None:

        rating_column = st.selectbox(
            "⭐ Rating Column",
            df.columns
        )

    else:

        st.info(
            f"⭐ Rating: **{rating_column}**"
        )


with col3:

    if category_column is None:

        category_column = st.selectbox(
            "📂 Category Column",
            df.columns
        )

    else:

        st.info(
            f"📂 Category: **{category_column}**"
        )


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

df["Sentiment"] = df[
    feedback_column
].apply(analyze_sentiment)


# =========================================================
# RATING CLEANING
# =========================================================

df[rating_column] = pd.to_numeric(
    df[rating_column],
    errors="coerce"
)


# =========================================================
# KEY BUSINESS METRICS
# =========================================================

total_responses = len(df)

positive_count = (
    df["Sentiment"] == "Positive"
).sum()

neutral_count = (
    df["Sentiment"] == "Neutral"
).sum()

negative_count = (
    df["Sentiment"] == "Negative"
).sum()


positive_percentage = (
    positive_count / total_responses * 100
    if total_responses > 0 else 0
)

negative_percentage = (
    negative_count / total_responses * 100
    if total_responses > 0 else 0
)

average_rating = df[
    rating_column
].mean()


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📈 Executive Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Total Responses",
        f"{total_responses:,}"
    )


with m2:

    st.metric(
        "Average Rating",
        f"{average_rating:.2f} ⭐"
        if pd.notna(average_rating)
        else "N/A"
    )


with m3:

    st.metric(
        "Positive Feedback",
        f"{positive_percentage:.1f}%"
    )


with m4:

    st.metric(
        "Negative Feedback",
        f"{negative_percentage:.1f}%"
    )


# =========================================================
# SENTIMENT + RATING
# =========================================================

st.markdown(
    '<div class="section-title">😊 Customer Sentiment & Ratings</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ---------- SENTIMENT ----------

with col1:

    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig_sentiment = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        hole=0.45,
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22c55e",
            "Neutral": "#94a3b8",
            "Negative": "#ef4444"
        },
        title="Overall Customer Sentiment"
    )

    fig_sentiment.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )


# ---------- RATING ----------

with col2:

    rating_counts = (
        df[rating_column]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_counts.columns = [
        "Rating",
        "Count"
    ]

    fig_rating = px.bar(
        rating_counts,
        x="Rating",
        y="Count",
        text="Count",
        title="Customer Rating Distribution"
    )

    fig_rating.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )


# =========================================================
# CATEGORY PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">📂 Category Performance</div>',
    unsafe_allow_html=True
)

category_rating = (
    df.groupby(category_column)[rating_column]
    .mean()
    .reset_index()
    .sort_values(
        rating_column,
        ascending=False
    )
)


fig_category = px.bar(
    category_rating,
    x=category_column,
    y=rating_column,
    text_auto=".2f",
    title="Average Customer Rating by Category"
)

fig_category.update_layout(
    template="plotly_white",
    xaxis_tickangle=-35
)

st.plotly_chart(
    fig_category,
    use_container_width=True
)


# =========================================================
# CATEGORY + SENTIMENT
# =========================================================

category_sentiment = pd.crosstab(
    df[category_column],
    df["Sentiment"]
)


fig_category_sentiment = px.bar(
    category_sentiment,
    barmode="group",
    title="Sentiment Distribution by Category"
)

fig_category_sentiment.update_layout(
    template="plotly_white"
)

st.plotly_chart(
    fig_category_sentiment,
    use_container_width=True
)


# =========================================================
# NEGATIVE FEEDBACK
# =========================================================

st.markdown(
    '<div class="section-title">🚨 Priority Negative Feedback</div>',
    unsafe_allow_html=True
)

negative_feedback = df[
    df["Sentiment"] == "Negative"
]


if len(negative_feedback) > 0:

    st.dataframe(
        negative_feedback[
            [
                category_column,
                rating_column,
                feedback_column
            ]
        ].head(10),
        use_container_width=True
    )

else:

    st.success(
        "🎉 No negative feedback detected."
    )


# =========================================================
# AUTOMATED BUSINESS INSIGHTS
# =========================================================

st.markdown(
    '<div class="section-title">💡 Automated Business Insights</div>',
    unsafe_allow_html=True
)


if len(df) > 0:

    most_common_category = (
        df[category_column]
        .value_counts()
        .idxmax()
    )

    category_negative = (
        negative_feedback[category_column]
        .value_counts()
    )

    if len(category_negative) > 0:

        most_negative_category = (
            category_negative.idxmax()
        )

    else:

        most_negative_category = "None"


    if positive_percentage >= 60:

        overall_sentiment = "Positive 😊"

    elif negative_percentage >= 40:

        overall_sentiment = "Needs Attention 🚨"

    else:

        overall_sentiment = "Mixed / Neutral 😐"


    st.info(
        f"""
        **Executive Summary**

        • Total customer responses: **{total_responses}**

        • Average customer rating: **{average_rating:.2f}**

        • Positive feedback: **{positive_percentage:.1f}%**

        • Negative feedback: **{negative_percentage:.1f}%**

        • Most common feedback category: **{most_common_category}**

        • Category with most negative feedback: **{most_negative_category}**

        • Overall customer sentiment: **{overall_sentiment}**
        """
    )


# =========================================================
# FILTER FEEDBACK
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Explore Customer Feedback</div>',
    unsafe_allow_html=True
)


filter_choice = st.selectbox(
    "Select sentiment to view:",
    [
        "All",
        "Positive",
        "Neutral",
        "Negative"
    ]
)


if filter_choice == "All":

    filtered_df = df

else:

    filtered_df = df[
        df["Sentiment"] == filter_choice
    ]


st.dataframe(
    filtered_df[
        [
            category_column,
            rating_column,
            feedback_column,
            "Sentiment"
        ]
    ],
    use_container_width=True
)


# =========================================================
# DOWNLOAD RESULTS
# =========================================================

st.markdown(
    '<div class="section-title">📥 Export Results</div>',
    unsafe_allow_html=True
)


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Analyzed Survey Data",
    data=csv_data,
    file_name="analyzed_survey_results.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Survey Intelligence Tool | "
    "Python • Pandas • Streamlit • Plotly • NLTK"
)
