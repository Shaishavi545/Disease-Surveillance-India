import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="India Disease Surveillance Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    df['Date'] = pd.to_datetime(df['Year_Week'] + '-1', format='%Y-W%W-%w')
    return df

df = load_data()

# ---------------- CLEANING ----------------
df['disease_clean'] = df['disease_clean'].fillna('Unknown').astype(str)
df['State'] = df['State'].fillna('Unknown').astype(str)
df['District'] = df['District'].fillna('Unknown').astype(str)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect("State", states)

diseases = sorted(df['disease_clean'].unique())
selected_diseases = st.sidebar.multiselect("Disease", diseases)

months = list(range(1, 13))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
selected_months = st.sidebar.multiselect(
    "Month",
    months,
    default=months,
    format_func=lambda x: month_names[x-1]
)

# ---------------- FILTER ----------------
filtered_df = df[df['Year'].isin(selected_years)]
filtered_df = filtered_df[filtered_df['Month_Approx'].isin(selected_months)]

if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
if selected_diseases:
    filtered_df = filtered_df[filtered_df['disease_clean'].isin(selected_diseases)]

# ---------------- HEADER ----------------
st.title("India Disease Surveillance Dashboard")
st.caption("Exploratory Analysis and Anomaly Detection (2020–2025)")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Trends",
    "Geographic",
    "Disease Analysis",
    "Seasonality",
    "Anomaly Detection"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================
with tab1:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", f"{filtered_df['Cases'].sum():,}")
    col2.metric("Reports", f"{len(filtered_df):,}")
    col3.metric("Diseases", filtered_df['disease_clean'].nunique())
    col4.metric("States", filtered_df['State'].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Top diseases
    with col1:
        top_diseases = filtered_df.groupby('disease_clean')['Cases'].sum()\
            .sort_values(ascending=False).head(15).reset_index()

        fig = px.bar(top_diseases, x='Cases', y='disease_clean',
                     orientation='h', color='Cases',
                     title="Top Diseases by Cases")

        st.plotly_chart(fig, width='stretch')

    # Top states
    with col2:
        top_states = filtered_df.groupby('State')['Cases'].sum()\
            .sort_values(ascending=False).head(15).reset_index()

        fig = px.bar(top_states, x='Cases', y='State',
                     orientation='h', color='Cases',
                     title="Top States by Cases")

        st.plotly_chart(fig, width='stretch')

    # Pie charts
    col1, col2 = st.columns(2)

    with col1:
        disease_dist = filtered_df.groupby('disease_clean')['Cases'].sum()\
            .sort_values(ascending=False).head(10).reset_index()

        fig = px.pie(disease_dist, values='Cases', names='disease_clean',
                     title="Disease Distribution")

        st.plotly_chart(fig, width='stretch')

    with col2:
        state_dist = filtered_df.groupby('State')['Cases'].sum()\
            .sort_values(ascending=False).head(10).reset_index()

        fig = px.pie(state_dist, values='Cases', names='State',
                     title="State Distribution")

        st.plotly_chart(fig, width='stretch')


# =====================================================
# TAB 2: TRENDS
# =====================================================
with tab2:
    st.subheader("Temporal Trends")

    yearly = filtered_df.groupby('Year')['Cases'].sum().reset_index()

    fig = px.bar(yearly, x='Year', y='Cases', title="Cases by Year")
    st.plotly_chart(fig, width='stretch')

    monthly = filtered_df.groupby('Month_Approx')['Cases'].sum().reset_index()
    monthly['Month'] = monthly['Month_Approx'].map(lambda x: month_names[x-1])

    fig = px.line(monthly, x='Month', y='Cases', title="Seasonal Trend")
    st.plotly_chart(fig, width='stretch')


# =====================================================
# TAB 3: GEOGRAPHIC
# =====================================================
with tab3:
    st.subheader("Geographic Analysis")

    state_data = filtered_df.groupby('State')['Cases'].sum().reset_index()

    fig = px.bar(state_data.sort_values('Cases', ascending=False).head(20),
                 x='State', y='Cases', color='Cases',
                 title="Top States by Cases")

    st.plotly_chart(fig, width='stretch')

    st.dataframe(state_data.sort_values('Cases', ascending=False), width='stretch')


# =====================================================
# TAB 4: DISEASE ANALYSIS
# =====================================================
with tab4:
    st.subheader("Disease Deep Dive")

    selected_disease = st.selectbox("Select Disease", diseases)

    disease_df = filtered_df[filtered_df['disease_clean'] == selected_disease]

    col1, col2 = st.columns(2)

    with col1:
        timeline = disease_df.groupby('Date')['Cases'].sum().reset_index()

        fig = px.line(timeline, x='Date', y='Cases',
                      title=f"{selected_disease} Trend")

        st.plotly_chart(fig, width='stretch')

    with col2:
        state_split = disease_df.groupby('State')['Cases'].sum()\
            .sort_values(ascending=False).head(10).reset_index()

        fig = px.bar(state_split, x='Cases', y='State',
                     orientation='h',
                     title="Top States")

        st.plotly_chart(fig, width='stretch')


# =====================================================
# TAB 5: SEASONAL
# =====================================================
with tab5:
    st.subheader("Seasonal Patterns")

    heatmap_data = filtered_df.pivot_table(
        values='Cases',
        index='disease_clean',
        columns='Month_Approx',
        aggfunc='sum',
        fill_value=0
    )

    heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]

    fig = px.imshow(heatmap_data,
                    labels=dict(x="Month", y="Disease", color="Cases"),
                    aspect='auto')

    st.plotly_chart(fig, width='stretch')


# =====================================================
# TAB 6: ANOMALY DETECTION (YOUR CORE IDEA)
# =====================================================
with tab6:
    st.subheader("Mortality Anomaly Detection")

    # Baseline: diseases with zero deaths overall
    baseline = df.groupby('disease_clean')['Deaths'].sum().reset_index()
    non_lethal = baseline[baseline['Deaths'] == 0]['disease_clean']

    anomaly_df = filtered_df[
        (filtered_df['disease_clean'].isin(non_lethal)) &
        (filtered_df['Deaths'] > 0)
    ]

    anomaly_summary = anomaly_df.groupby(
        ['disease_clean', 'Year', 'Month_Approx']
    ).agg({
        'Cases': 'sum',
        'Deaths': 'sum'
    }).reset_index()

    anomaly_summary['CFR'] = (anomaly_summary['Deaths'] / anomaly_summary['Cases']) * 100

    st.markdown("""
    These events represent cases where deaths were reported for diseases that typically show no mortality.
    Such deviations may indicate data inconsistencies or unusual outbreak severity.
    """)

    if not anomaly_summary.empty:

        fig = px.bar(
            anomaly_summary.sort_values('CFR', ascending=False).head(10),
            x='CFR',
            y='disease_clean',
            color='Cases',
            orientation='h',
            title="Top Anomalous Events"
        )

        st.plotly_chart(fig, width='stretch')

        # Drill-down
        selected_anomaly = st.selectbox(
            "Investigate Disease",
            anomaly_summary['disease_clean'].unique()
        )

        deep_df = anomaly_df[anomaly_df['disease_clean'] == selected_anomaly]

        fig = px.line(
            deep_df.groupby('Date')[['Cases', 'Deaths']].sum().reset_index(),
            x='Date',
            y=['Cases', 'Deaths'],
            title=f"{selected_anomaly} Anomaly Timeline"
        )

        st.plotly_chart(fig, width='stretch')

    else:
        st.info("No anomalies detected for selected filters.")