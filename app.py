import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="ADSSU Circular Economy Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {

    min-height: 100vh !important;

    overflow-x: hidden !important;

    background:

        radial-gradient(
            circle at 15% 15%,
            rgba(187, 247, 208, 0.45) 0%,
            transparent 25%
        ),

        radial-gradient(
            circle at 85% 20%,
            rgba(74, 222, 128, 0.35) 0%,
            transparent 28%
        ),

        radial-gradient(
            circle at 70% 85%,
            rgba(16, 185, 129, 0.30) 0%,
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #d9f7df 0%,
            #a7dfb3 30%,
            #7bcf91 60%,
            #4caf78 100%
        ) !important;

    background-attachment: fixed !important;

}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container {

    background: transparent !important;

}

header[data-testid="stHeader"] {

    background: rgba(0, 0, 0, 0) !important;

    box-shadow: none !important;

}

.stApp::before {

    content: "";

    position: fixed;

    width: 520px;
    height: 520px;

    top: -220px;
    right: -160px;

    border-radius: 50%;

    background:

        radial-gradient(
            circle at 35% 30%,
            rgba(220, 252, 231, 0.70),
            rgba(74, 222, 128, 0.35) 38%,
            rgba(22, 163, 74, 0.15) 62%,
            transparent 72%
        );

    box-shadow:

        inset -40px -40px 80px
        rgba(20, 83, 45, 0.18),

        inset 25px 25px 60px
        rgba(255,255,255,0.55),

        0 40px 100px
        rgba(20,83,45,0.18);

    opacity: 0.8;

    pointer-events: none;

    z-index: 0;

}

.stApp::after {

    content: "";

    position: fixed;

    width: 420px;
    height: 420px;

    bottom: -180px;
    left: -160px;

    border-radius: 50%;

    background:

        radial-gradient(
            circle at 40% 30%,
            rgba(220,252,231,0.65),
            rgba(34,197,94,0.30) 40%,
            rgba(21,128,61,0.12) 65%,
            transparent 72%
        );

    box-shadow:

        inset -30px -30px 70px
        rgba(20,83,45,0.18),

        inset 20px 20px 50px
        rgba(255,255,255,0.55),

        0 30px 80px
        rgba(20,83,45,0.16);

    opacity: 0.8;

    pointer-events: none;

    z-index: 0;

}

/*
   IMPORTANT:
   CSS-only animation.
   No <span>
   No <div>
   No inline style
   No --duration
   No --delay
*/


.stApp {

    --leaf1: "🍃";
    --leaf2: "🍂";
    --leaf3: "♻️";
    --leaf4: "🧴";
    --leaf5: "🥫";

}

[data-testid="stAppViewContainer"]::before {

    content:
        "🍃  🍂  🍃  ♻️  🍂  🍃  🧴  🍂  🍃  ♻️";
                    
    position: fixed;

    top: -100px;
    left: -5%;

    width: 110%;

    height: 100vh;

    font-size: 30px;

    letter-spacing: 45px;

    word-spacing: 40px;

    line-height: 120px;

    white-space: normal;

    pointer-events: none;

    z-index: 1;

    opacity: 100;

    filter:
        drop-shadow(
            0 10px 8px
            rgba(20,83,45,0.25)
        );

    animation:
        fallingLayerOne
        22s
        linear
        infinite;

}

[data-testid="stMain"]::before {

    content:
        "🍂    🍃    📄    🍃    🥫    🍂    ♻️    🍃    🧃    🍂";

    position: fixed;

    top: -150px;
    left: -10%;

    width: 120%;

    height: 100vh;

    font-size: 24px;

    letter-spacing: 55px;

    word-spacing: 30px;

    line-height: 140px;

    white-space: normal;

    pointer-events: none;

    z-index: 1;

    opacity: 100;

    filter:
        drop-shadow(
            0 9px 7px
            rgba(20,83,45,0.22)
        );

    animation:
        fallingLayerTwo
        28s
        linear
        infinite;

}

[data-testid="stAppViewContainer"]::after {

    content:
        "🍃       🍂       🍃       🧴       🍂       ♻️";

    position: fixed;

    top: -200px;

    left: 5%;

    width: 100%;

    height: 100vh;

    font-size: 20px;

    letter-spacing: 70px;

    word-spacing: 55px;

    line-height: 180px;

    pointer-events: none;

    z-index: 1;

    opacity: 100;

    filter:
        drop-shadow(
            0 8px 6px
            rgba(20,83,45,0.20)
        );

    animation:
        fallingLayerThree
        50s
        linear
        infinite;

}

@keyframes fallingLayerOne {

    0% {

        transform:
            translate3d(
                -30px,
                -150px,
                0
            )
            rotate(0deg);

        opacity: 0;

    }

    10% {

        opacity: 0.42;

    }

    25% {

        transform:
            translate3d(
                80px,
                25vh,
                0
            )
            rotate(80deg);

    }

    50% {

        transform:
            translate3d(
                -80px,
                50vh,
                0
            )
            rotate(180deg);

    }

    75% {

        transform:
            translate3d(
                100px,
                75vh,
                0
            )
            rotate(280deg);

    }

    100% {

        transform:
            translate3d(
                -60px,
                115vh,
                0
            )
            rotate(380deg);

        opacity: 0;

    }

}

@keyframes fallingLayerTwo {

    0% {

        transform:
            translate3d(
                60px,
                -180px,
                0
            )
            rotate(0deg);

        opacity: 0;

    }

    12% {

        opacity: 0.30;

    }

    30% {

        transform:
            translate3d(
                -100px,
                30vh,
                0
            )
            rotate(-100deg);

    }

    55% {

        transform:
            translate3d(
                120px,
                55vh,
                0
            )
            rotate(190deg);

    }

    80% {

        transform:
            translate3d(
                -80px,
                80vh,
                0
            )
            rotate(290deg);

    }

    100% {

        transform:
            translate3d(
                80px,
                115vh,
                0
            )
            rotate(400deg);

        opacity: 0;

    }

}

@keyframes fallingLayerThree {

    0% {

        transform:
            translate3d(
                -80px,
                -200px,
                0
            )
            rotate(0deg)
            scale(0.8);

        opacity: 0;

    }

    15% {

        opacity: 0.25;

    }

    35% {

        transform:
            translate3d(
                110px,
                35vh,
                0
            )
            rotate(120deg)
            scale(1);

    }

    60% {

        transform:
            translate3d(
                -120px,
                60vh,
                0
            )
            rotate(230deg)
            scale(1.1);

    }

    85% {

        transform:
            translate3d(
                80px,
                85vh,
                0
            )
            rotate(320deg)
            scale(0.9);

    }

    100% {

        transform:
            translate3d(
                -50px,
                115vh,
                0
            )
            rotate(420deg)
            scale(0.7);

        opacity: 0;

    }

}

[data-testid="stMainBlockContainer"] {

    position: relative;

    z-index: 5;

}

section[data-testid="stSidebar"] {

    position: relative;

    z-index: 10;

    background:

        linear-gradient(
            145deg,
            rgba(220,252,231,0.96),
            rgba(166,230,180,0.94)
        ) !important;

    border-right:

        1px solid
        rgba(21,128,61,0.25);

    box-shadow:

        10px 0 35px
        rgba(20,83,45,0.18),

        inset -2px 0
        rgba(255,255,255,0.70);

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

}

.nav-title {

    font-size: 23px;

    font-weight: 850;

    color: #064e3b;

    letter-spacing: -0.5px;

    text-shadow:

        1px 2px 3px
        rgba(20,83,45,0.20);

}


.main-title {

    font-size: 34px;

    font-weight: 850;

    color: #064e3b;

    letter-spacing: -1px;

    text-shadow:

        2px 3px 6px
        rgba(20,83,45,0.20);

}


.subtitle {

    color: #285943;

    font-size: 15px;

    font-weight: 600;

}


.section-title {

    font-size: 27px;

    font-weight: 800;

    color: #064e3b;

    text-shadow:

        1px 2px 4px
        rgba(20,83,45,0.18);

}

.card {

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.90),
            rgba(220,252,231,0.82)
        );

    padding: 24px;

    border-radius: 22px;

    border:

        1px solid
        rgba(255,255,255,0.75);

    margin-bottom: 20px;

    box-shadow:

        0 18px 40px
        rgba(20,83,45,0.18),

        0 5px 12px
        rgba(20,83,45,0.10),

        inset 0 1px 0
        rgba(255,255,255,0.95);

    backdrop-filter:
        blur(12px);

    -webkit-backdrop-filter:
        blur(12px);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;

}


.card:hover {

    transform:

        translateY(-6px)
        scale(1.01);

    box-shadow:

        0 25px 50px
        rgba(20,83,45,0.22);

}

div[data-testid="stMetric"] {

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.94),
            rgba(220,252,231,0.86)
        );

    border-radius: 20px;

    padding: 20px;

    border:

        1px solid
        rgba(255,255,255,0.80);

    box-shadow:

        0 15px 30px
        rgba(20,83,45,0.16),

        inset 0 1px 0
        rgba(255,255,255,0.95);

    backdrop-filter:
        blur(12px);

    -webkit-backdrop-filter:
        blur(12px);

}


div[data-testid="stMetric"]:hover {

    transform:
        translateY(-5px);

    box-shadow:

        0 24px 45px
        rgba(20,83,45,0.22);

}


div[data-testid="stMetricLabel"] {

    color: #315b42;

    font-weight: 700;

}


div[data-testid="stMetricValue"] {

    color: #064e3b;

    font-weight: 850;

    text-shadow:

        1px 3px 5px
        rgba(20,83,45,0.15);

}

div[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border:

        1px solid
        rgba(255,255,255,0.70);

    box-shadow:

        0 15px 30px
        rgba(20,83,45,0.14);

}

[data-testid="stFileUploader"] {

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.92),
            rgba(214,249,222,0.90)
        );

    border-radius: 18px;

    padding: 8px;

    border:

        1px solid
        rgba(255,255,255,0.75);

    box-shadow:

        0 12px 25px
        rgba(20,83,45,0.14);

}

.stButton > button {

    border-radius: 14px;

    border:

        1px solid
        rgba(34,197,94,0.35);

    background:

        linear-gradient(
            145deg,
            #22c55e,
            #15803d
        );

    color: white;

    font-weight: 750;

    box-shadow:

        0 8px 15px
        rgba(21,128,61,0.25),

        inset 0 1px 0
        rgba(255,255,255,0.35);

}


.stButton > button:hover {

    transform:

        translateY(-3px);

    box-shadow:

        0 14px 25px
        rgba(21,128,61,0.30);

}

section[data-testid="stSidebar"]
div[role="radiogroup"] label {

    border-radius: 12px;

    padding: 8px 10px;

    margin: 4px 0;

    transition:
        background 0.2s ease,
        transform 0.2s ease;

}


section[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {

    background:
        rgba(34,197,94,0.18);

    transform:
        translateX(4px);

}

div[data-baseweb="select"] > div {

    border-radius: 13px !important;

    border:

        1px solid
        rgba(21,128,61,0.25);

    background:
        rgba(255,255,255,0.90);

}

div[data-testid="stAlert"] {

    border-radius: 16px;

    box-shadow:

        0 10px 22px
        rgba(20,83,45,0.10);

    backdrop-filter:
        blur(8px);

}

hr {

    border: none;

    height: 2px;

    background:

        linear-gradient(
            90deg,
            transparent,
            rgba(21,128,61,0.35),
            transparent
        );

}

::-webkit-scrollbar {

    width: 9px;

}


::-webkit-scrollbar-track {

    background:
        #b7e4c7;

}


::-webkit-scrollbar-thumb {

    background:

        linear-gradient(
            180deg,
            #52b788,
            #15803d
        );

    border-radius: 20px;

}

@media (prefers-reduced-motion: reduce) {

    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after,
    [data-testid="stMain"]::before {

        animation: none !important;

    }

}

</style>
""", unsafe_allow_html=True)
    

def read_dataset(file):

    if file.name.lower().endswith(".csv"):

        try:
            df = pd.read_csv(file, encoding="utf-8-sig")

        except UnicodeDecodeError:
            file.seek(0)

            try:
                df = pd.read_csv(file, encoding="cp1252")

            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(
                    file,
                    encoding="latin1"
                )

    elif file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format.")

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Clean student names without converting them to numbers
    name_columns = [
        column for column in df.columns
        if column.strip().lower() in [
            "name",
            "student name",
            "student_name",
            "full name",
            "fullname",
            "student"
        ]
    ]

    for column in name_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


def numeric_data(df, columns):

    return df[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )


def interpret_mean(value):

    if pd.isna(value):
        return "No Data"

    if value >= 4.21:
        return "Very High"

    elif value >= 3.41:
        return "High"

    elif value >= 2.61:
        return "Moderate"

    elif value >= 1.81:
        return "Low"

    else:
        return "Very Low"


def practice_interpretation(value):

    if pd.isna(value):
        return "No Data"

    if value >= 4.21:
        return "Always"

    elif value >= 3.41:
        return "Often"

    elif value >= 2.61:
        return "Sometimes"

    elif value >= 1.81:
        return "Rarely"

    else:
        return "Never"


st.sidebar.markdown(
    '<div class="nav-title">♻️ ADSSU Dashboard</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "Circular Economy Data Mining System"
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    try:

        df = read_dataset(uploaded_file)

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        st.sidebar.success(
            f"Dataset loaded: {len(df):,} respondents"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error reading dataset: {e}"
        )

        st.stop()

else:

    df = None

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 📌 Navigation"
)

page = st.sidebar.radio(
    "Select Analysis",
    [
        "🏠 Overview",
        "📂 Dataset",
        "👤 Demographics",
        "🧠 Weighted Mean",
        "🔵 K-Means",
        "🔗 Apriori",
        "📋 Policy Framework"
    ]
)

if df is None:

    st.markdown(
        '<div class="main-title">♻️ Circular Economy Student Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Agusan del Sur State University (ADSSU) – Main Campus</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.info(
        "📂 Please upload your survey dataset using the "
        "Upload Dataset button in the sidebar."
    )

    st.markdown("""
    <div class="card">

    <h2>WELCOME</h2>

    <p>
    This dashboard analyzes the Circular Economy awareness
    and sustainability practices of students using data mining
    techniques.
    </p>

    <br>

    <b>Available Analysis:</b>

    <ul>
        <li>👤 Demographic Profile</li>
        <li>🧠 Weighted Mean</li>
        <li>🔵 K-Means Clustering</li>
        <li>🔗 Apriori Association Rules</li>
        <li>📋 Data-Driven Policy Framework</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.stop()

all_columns = df.columns.tolist()


numeric_columns = (
    df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

# Exclude student name columns from numeric analysis
name_columns = [
    column for column in df.columns
    if column.strip().lower() in [
        "name",
        "student name",
        "student_name",
        "full name",
        "fullname",
        "student"
    ]
]

numeric_columns = [
    column for column in numeric_columns
    if column not in name_columns
]

awareness_columns = []
practice_columns = []

age_column = "Not Selected"
gender_column = "Not Selected"
college_column = "Not Selected"
year_column = "Not Selected"

if page == "👤 Demographics":

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Demographic Columns")


    def find_column(keywords):

        for keyword in keywords:

            for column in all_columns:

                if keyword.lower() in column.lower():

                    return column

        return None


    age_guess = find_column(["age"])

    gender_guess = find_column(
        ["gender", "sex"]
    )

    college_guess = find_column(
        ["college"]
    )

    year_guess = find_column(
        ["year level", "year", "level"]
    )


    age_column = st.sidebar.selectbox(
        "Age",
        ["Not Selected"] + all_columns,
        index=(
            all_columns.index(age_guess) + 1
            if age_guess in all_columns
            else 0
        )
    )


    gender_column = st.sidebar.selectbox(
        "Sex / Gender",
        ["Not Selected"] + all_columns,
        index=(
            all_columns.index(gender_guess) + 1
            if gender_guess in all_columns
            else 0
        )
    )


    college_column = st.sidebar.selectbox(
        "College",
        ["Not Selected"] + all_columns,
        index=(
            all_columns.index(college_guess) + 1
            if college_guess in all_columns
            else 0
        )
    )


    year_column = st.sidebar.selectbox(
        "Year Level",
        ["Not Selected"] + all_columns,
        index=(
            all_columns.index(year_guess) + 1
            if year_guess in all_columns
            else 0
        )
    )

if page == "🧠 Weighted Mean":

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Awareness Items")

    awareness_columns = st.sidebar.multiselect(
        "Select Awareness Questions",
        numeric_columns
    )


    st.sidebar.markdown("### ♻️ Practice Items")

    practice_columns = st.sidebar.multiselect(
        "Select Practice Questions",
        numeric_columns
    )

if page == "🔵 K-Means":

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Awareness Items")

    awareness_columns = st.sidebar.multiselect(
        "Select Awareness Questions",
        numeric_columns
    )


    st.sidebar.markdown("### ♻️ Practice Items")

    practice_columns = st.sidebar.multiselect(
        "Select Practice Questions",
        numeric_columns
    )

if page == "🔗 Apriori":

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🧠 Awareness Items")
    awareness_columns = st.sidebar.multiselect(
        "Select Awareness Questions",
        numeric_columns,
        key="apriori_awareness"
    )

    st.sidebar.markdown("### ♻️ Practice Items")
    practice_columns = st.sidebar.multiselect(
        "Select Practice Questions",
        numeric_columns,
        key="apriori_practices"
    )

if page == "📋 Policy Framework":

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Awareness Items")

    awareness_columns = st.sidebar.multiselect(
        "Select Awareness Questions",
        numeric_columns
    )


    st.sidebar.markdown("### ♻️ Practice Items")

    practice_columns = st.sidebar.multiselect(
        "Select Practice Questions",
        numeric_columns
    )

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">Circular Economy Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Agusan del Sur State University (ADSSU) – Main Campus</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "##  Dashboard Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            " Respondents",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            " Variables",
            f"{len(df.columns):,}"
        )

    with col3:

        st.metric(
            " Awareness Items",
            len(awareness_columns)
        )

    with col4:

        st.metric(
            " Practice Items",
            len(practice_columns)
        )


    st.markdown("###  Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### 🧹 Data Quality")

    missing = df.isna().sum().sum()

    if missing == 0:

        st.success(
            "No missing values detected."
        )

    else:

        st.warning(
            f"{missing:,} missing cells detected."
        )

elif page == "📂 Dataset":

    st.markdown(
        '<div class="section-title">📂 Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Rows:** {len(df):,}"
    )

    st.write(
        f"**Columns:** {len(df.columns):,}"
    )


    st.markdown("### 📋 Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### 🧹 Missing Values")

    missing_table = pd.DataFrame({

        "Column":
            df.columns,

        "Missing":
            df.isna().sum().values,

        "Missing Percentage":
            (
                df.isna().mean() * 100
            ).round(2).values

    })


    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

elif page == "👤 Demographics":

    st.markdown(
        '<div class="section-title">👤 Respondent Demographic Profile</div>',
        unsafe_allow_html=True
    )


    demographic_columns = [

        ("Age", age_column),

        ("Sex / Gender", gender_column),

        ("College", college_column),

        ("Year Level", year_column)

    ]


    for label, column in demographic_columns:

        if column == "Not Selected":
            continue


        st.markdown(
            f"### {label}"
        )


        frequency = (
            df[column]
            .value_counts(dropna=False)
        )


        percentage = (
            df[column]
            .value_counts(
                normalize=True,
                dropna=False
            )
            * 100
        )


        result = pd.DataFrame({

            "Frequency":
                frequency,

            "Percentage":
                percentage.round(2)

        })


        result.index = (
            result.index.astype(str)
        )


        st.dataframe(
            result,
            use_container_width=True
        )


        fig, ax = plt.subplots(
            figsize=(9, 4)
        )


        frequency.plot(
            kind="bar",
            ax=ax
        )


        ax.set_title(
            f"{label} Distribution"
        )

        ax.set_ylabel(
            "Frequency"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

elif page == "🧠 Weighted Mean":

    st.markdown(
        '<div class="section-title"> Circular Economy Weighted Mean</div>',
        unsafe_allow_html=True
    )


    if not awareness_columns and not practice_columns:

        st.warning(
            "Please select Awareness or Practice questions "
            "from the sidebar."
        )

    else:

        if awareness_columns:

            st.markdown(
                "###  Awareness Results"
            )


            awareness = numeric_data(
                df,
                awareness_columns
            )


            awareness_means = (
                awareness.mean()
            )


            overall_awareness = (
                awareness.stack().mean()
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Overall Awareness Mean",
                    f"{overall_awareness:.2f}"
                )


            with col2:

                st.metric(
                    "Interpretation",
                    interpret_mean(
                        overall_awareness
                    )
                )


            awareness_result = pd.DataFrame({

                "Question":
                    awareness_means.index,

                "Weighted Mean":
                    awareness_means.round(2).values,

                "Interpretation":
                    awareness_means.apply(
                        interpret_mean
                    ).values

            })


            st.dataframe(
                awareness_result,
                use_container_width=True,
                hide_index=True
            )

        if practice_columns:

            st.markdown(
                "### ♻️ Sustainability Practice Results"
            )


            practices = numeric_data(
                df,
                practice_columns
            )


            practice_means = (
                practices.mean()
            )


            overall_practice = (
                practices.stack().mean()
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Overall Practice Mean",
                    f"{overall_practice:.2f}"
                )


            with col2:

                st.metric(
                    "Interpretation",
                    practice_interpretation(
                        overall_practice
                    )
                )


            practice_result = pd.DataFrame({

                "Question":
                    practice_means.index,

                "Weighted Mean":
                    practice_means.round(2).values,

                "Interpretation":
                    practice_means.apply(
                        practice_interpretation
                    ).values

            })


            st.dataframe(
                practice_result,
                use_container_width=True,
                hide_index=True
            )

elif page == "🔵 K-Means":

    st.markdown(
        '<div class="section-title">🔵 K-Means Student Clustering</div>',
        unsafe_allow_html=True
    )


    st.write(
        """
        K-Means groups respondents based on similarities
        in their Circular Economy awareness and sustainability practices.
        """
    )


    if (
        not awareness_columns
        or not practice_columns
    ):

        st.warning(
            "Please select both Awareness and Practice questions "
            "from the sidebar."
        )

    else:

        awareness_data = numeric_data(
            df,
            awareness_columns
        )

        practice_data = numeric_data(
            df,
            practice_columns
        )


        features = pd.DataFrame({

            "Awareness":
                awareness_data.mean(axis=1),

            "Practices":
                practice_data.mean(axis=1)

        })


        features = features.dropna()


        if len(features) < 3:

            st.error(
                "Not enough complete respondents for K-Means."
            )

        else:

            scaler = StandardScaler()

            scaled = scaler.fit_transform(
                features
            )


            model = KMeans(

                n_clusters=3,

                random_state=42,

                n_init=20

            )


            features["Cluster"] = (
                model.fit_predict(scaled)
            )


            # Determine cluster order
            cluster_means = (
                features
                .groupby("Cluster")
                [["Awareness", "Practices"]]
                .mean()
            )


            cluster_means["Overall"] = (
                cluster_means.mean(axis=1)
            )


            ordered = (
                cluster_means["Overall"]
                .sort_values()
                .index.tolist()
            )


            names = {

                ordered[0]:
                    "Low",

                ordered[1]:
                    "Moderate",

                ordered[2]:
                    "High"

            }


            features["Cluster Level"] = (
                features["Cluster"]
                .map(names)
            )


            st.markdown(
                "###  Cluster Distribution"
            )


            low = (
                features["Cluster Level"]
                .value_counts()
                .get("Low", 0)
            )


            moderate = (
                features["Cluster Level"]
                .value_counts()
                .get("Moderate", 0)
            )


            high = (
                features["Cluster Level"]
                .value_counts()
                .get("High", 0)
            )


            c1, c2, c3 = st.columns(3)


            with c1:
                st.metric(
                    "🔴 Low",
                    low
                )


            with c2:
                st.metric(
                    "🟡 Moderate",
                    moderate
                )


            with c3:
                st.metric(
                    "🟢 High",
                    high
                )


            st.markdown(
                "### 📋 Cluster Summary"
            )


            summary = (
                features
                .groupby("Cluster Level")
                [["Awareness", "Practices"]]
                .mean()
                .reindex(
                    [
                        "Low",
                        "Moderate",
                        "High"
                    ]
                )
                .round(2)
            )


            summary["Overall"] = (
                summary.mean(axis=1)
                .round(2)
            )


            st.dataframe(
                summary,
                use_container_width=True
            )


            st.markdown(
                "### 📍 Cluster Visualization"
            )


            fig, ax = plt.subplots(
                figsize=(9, 6)
            )


            for level in [
                "Low",
                "Moderate",
                "High"
            ]:

                subset = features[
                    features["Cluster Level"]
                    == level
                ]


                ax.scatter(
                    subset["Awareness"],
                    subset["Practices"],
                    label=level,
                    alpha=0.7
                )


            ax.set_xlabel(
                "Awareness"
            )

            ax.set_ylabel(
                "Sustainability Practices"
            )

            ax.set_title(
                "K-Means Student Clusters"
            )

            ax.legend()

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


elif page == "🔗 Apriori":

    st.markdown(
        '<div class="section-title">Apriori Association Rule Mining</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Apriori identifies frequently occurring combinations
        of sustainability practices and discovers relationships
        between them.
        """
    )

    if not practice_columns:
        st.warning("Please select Practice questions from the sidebar.")

    else:
        practice_data = numeric_data(df, practice_columns)

        st.markdown("### ⚙️ Apriori Settings")

        threshold = st.slider(
            "Practice Threshold",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.5
        )

        minimum_support = st.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=0.80,
            value=0.20,
            step=0.01
        )

        minimum_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05
        )

        binary = (
            practice_data >= threshold
        ).fillna(False).astype(bool)

        try:
            with st.spinner("Analyzing sustainability practices..."):

                itemsets = apriori(
                    binary,
                    min_support=minimum_support,
                    use_colnames=True,
                    max_len=3,
                    low_memory=True
                )

                if itemsets.empty:
                    st.warning(
                        "No frequent itemsets found. "
                        "Try lowering Minimum Support."
                    )

                else:
                    st.markdown("### Frequent Itemsets")

                    display_itemsets = itemsets.copy()

                    display_itemsets["Itemsets"] = (
                        display_itemsets["itemsets"].apply(
                            lambda x: ", ".join(str(i) for i in x)
                        )
                    )

                    display_itemsets = (
                        display_itemsets[
                            ["Itemsets", "support"]
                        ]
                        .sort_values(
                            "support",
                            ascending=False
                        )
                        .round(3)
                    )

                    st.dataframe(
                        display_itemsets,
                        use_container_width=True,
                        hide_index=True
                    )

                    rules = association_rules(
                        itemsets,
                        metric="confidence",
                        min_threshold=minimum_confidence
                    )

                    if rules.empty:
                        st.warning(
                            "No association rules found. "
                            "Try lowering Minimum Confidence."
                        )

                    else:
                        rules["Antecedents"] = (
                            rules["antecedents"].apply(
                                lambda x: ", ".join(str(i) for i in x)
                            )
                        )

                        rules["Consequents"] = (
                            rules["consequents"].apply(
                                lambda x: ", ".join(str(i) for i in x)
                            )
                        )

                        display_rules = (
                            rules[
                                [
                                    "Antecedents",
                                    "Consequents",
                                    "support",
                                    "confidence",
                                    "lift"
                                ]
                            ]
                            .sort_values(
                                "lift",
                                ascending=False
                            )
                            .round(3)
                        )

                        st.markdown("### 🔗 Association Rules")

                        st.dataframe(
                            display_rules,
                            use_container_width=True,
                            hide_index=True
                        )

                    st.markdown("### 📖 Interpretation")

                    st.info(
                        """
                        **Support** – frequency of the combination
                        in the dataset.

                        **Confidence** – likelihood that the consequent
                        occurs when the antecedent occurs.

                        **Lift** – strength of the relationship.
                        A lift greater than 1 suggests a positive association.
                        """
                    )

        except Exception as e:
            st.error(f"Apriori error: {e}")


elif page == "📋 Policy Framework":

    st.markdown(
        '<div class="section-title">📋 Data-Driven Policy Framework</div>',
        unsafe_allow_html=True
    )


    if (
        not awareness_columns
        or not practice_columns
    ):

        st.warning(
            "Please select Awareness and Practice questions first."
        )

    else:

        awareness = numeric_data(
            df,
            awareness_columns
        )


        practices = numeric_data(
            df,
            practice_columns
        )


        awareness_mean = (
            awareness.stack().mean()
        )


        practice_mean = (
            practices.stack().mean()
        )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                " Awareness",
                f"{awareness_mean:.2f}"
            )


        with c2:

            st.metric(
                "♻️ Practices",
                f"{practice_mean:.2f}"
            )


        st.markdown(
            "### 🎯 Suggested Areas for Intervention"
        )


        if awareness_mean < 3.41:

            st.warning(
                """
                **Awareness Intervention**

                Strengthen Circular Economy education,
                information campaigns, seminars, and
                sustainability learning activities.
                """
            )

        else:

            st.success(
                """
                **Awareness Strength**

                Student awareness is relatively strong.
                Existing educational initiatives may be maintained
                and continuously improved.
                """
            )


        if practice_mean < 3.41:

            st.warning(
                """
                **Practice Intervention**

                Strengthen programs that translate awareness
                into actual sustainable behaviors such as
                reuse, recycling, waste reduction, and
                responsible resource use.
                """
            )

        else:

            st.success(
                """
                **Practice Strength**

                Sustainable practices are relatively strong.
                Existing initiatives may be sustained and expanded.
                """
            )


        st.markdown(
            "###  Recommended Policy Areas"
        )


        p1, p2 = st.columns(2)


        with p1:

            st.markdown(
                """
                ####  1. Circular Economy Education

                Integrate Circular Economy concepts into
                student learning activities and environmental
                education programs.
                """
            )


            st.markdown(
                """
                #### 2. Waste Reduction

                Strengthen recycling, reuse, waste segregation,
                and waste minimization initiatives.
                """
            )


        with p2:

            st.markdown(
                """
                ####  3. Student Engagement

                Encourage students to participate in
                sustainability campaigns and activities.
                """
            )


            st.markdown(
                """
                ####  4. Data-Based Planning

                Use the dashboard findings to support
                evidence-based sustainability planning.
                """
            )

st.sidebar.markdown("---")

st.sidebar.caption(
    "♻️ ADSSU Circular Economy Data Mining Dashboard"
)
