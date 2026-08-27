import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] h2 {
        font-size: 1.25rem;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #64748b;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        position: relative;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.72rem 0.85rem;
        margin: 0.22rem 0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.035);
        transition: all 0.15s ease-in-out;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        border-color: #99f6e4;
        background: #f0fdfa;
        transform: translateY(-1px);
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3) {
        margin-top: 1.25rem;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3)::before {
        content: "";
        position: absolute;
        top: -0.75rem;
        left: 0;
        right: 0;
        height: 1px;
        background: #cbd5e1;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        border-color: #0f766e;
        background: #ecfdf5;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.12);
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #0f766e;
        font-weight: 750;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] p {
        color: #334155;
        font-weight: 620;
    }

    /* Header */
    .main-header {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 6px solid #0f766e;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.15;
        color: #0f172a;
        font-weight: 800;
    }

    .main-header p {
        margin-top: 12px;
        margin-bottom: 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        color: #0f766e;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        color: #111827;
        font-weight: 600;
        padding: 0.75rem 1rem;
        transition: all 0.15s ease-in-out;
    }

    div.stButton > button:hover {
        border-color: #0f766e;
        color: #0f766e;
        background: #f0fdfa;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 650;
        color: #0f172a;
    }

    /* Pipeline metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 136px;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
        overflow: visible;
    }

    [data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 650;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        line-height: 1.25;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 780;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        line-height: 1.15;
        font-size: 1.42rem;
        overflow-wrap: anywhere;
    }

    [data-testid="stMetric"] + div,
    [data-testid="stCaptionContainer"] {
        white-space: normal;
        overflow-wrap: anywhere;
    }

    /* Analysis selection cards */
    .analysis-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 16px 14px 16px;
        min-height: 112px;
        margin-bottom: 10px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
    }

    .analysis-card-title {
        color: #0f172a;
        font-size: 1.02rem;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .analysis-card-caption {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.4;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Remove old empty card style if it exists */
    .action-card {
        display: none !important;
    }
</style>
""",
        unsafe_allow_html=True,
    )
