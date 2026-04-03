import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="India Disease Surveillance Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    # Convert Year_Week to datetime for better plotting
    df['Date'] = pd.to_datetime(df['Year_Week'] + '-1', format='%Y-W%W-%w')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=years,
    default=years
)

# State filter
states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=states,
    default=[]  # Empty by default for performance
)

# Disease filter
diseases = sorted(df['disease_clean'].unique())
selected_diseases = st.sidebar.multiselect(
    "Select Disease(s)",
    options=diseases,
    default=[]
)

# Month filter
months = list(range(1, 13))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
selected_months = st.sidebar.multiselect(
    "Select Month(s)",
    options=months,
    format_func=lambda x: month_names[x-1],
    default=months
)

# District filter (conditional on state selection)
if selected_states:
    districts = sorted(df[df['State'].isin(selected_states)]['District'].dropna().unique())
    selected_districts = st.sidebar.multiselect(
        "Select District(s)",
        options=districts,
        default=[]
    )
else:
    selected_districts = []

# Apply filters
filtered_df = df[df['Year'].isin(selected_years)]
filtered_df = filtered_df[filtered_df['Month_Approx'].isin(selected_months)]

if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
if selected_diseases:
    filtered_df = filtered_df[filtered_df['disease_clean'].isin(selected_diseases)]
if selected_districts:
    filtered_df = filtered_df[filtered_df['District'].isin(selected_districts)]

# Main content
st.markdown('<p class="main-header">🦠 India Disease Surveillance Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data Quality Analysis & Outbreak Intelligence (2020-2025)</p>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "📈 Trends", 
    "🗺️ Geographic", 
    "🔍 Disease Deep Dive",
    "🌡️ Seasonal Patterns",
    "💡 Insights"
])

# ========================================
# TAB 1: OVERVIEW
# ========================================
with tab1:
    st.header("Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_cases = filtered_df['Cases'].sum()
        st.metric("Total Cases", f"{total_cases:,}")
    
    with col2:
        total_deaths = filtered_df['Deaths'].sum()
        st.metric("Total Deaths", f"{total_deaths:,}")
    
    with col3:
        cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0
        st.metric("CFR", f"{cfr:.2f}%")
    
    with col4:
        num_reports = len(filtered_df)
        st.metric("Reports", f"{num_reports:,}")
    
    with col5:
        num_diseases = filtered_df['disease_clean'].nunique()
        st.metric("Diseases", f"{num_diseases}")
    
    st.markdown("---")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Diseases by Cases")
        top_diseases = filtered_df.groupby('disease_clean')['Cases'].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=top_diseases.values,
            y=top_diseases.index,
            orientation='h',
            labels={'x': 'Total Cases', 'y': 'Disease'},
            color=top_diseases.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 15 States by Cases")
        top_states = filtered_df.groupby('State')['Cases'].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=top_states.values,
            y=top_states.index,
            orientation='h',
            labels={'x': 'Total Cases', 'y': 'State'},
            color=top_states.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Disease Distribution Pie Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Category Distribution")
        disease_dist = filtered_df.groupby('disease_clean')['Cases'].sum().sort_values(ascending=False).head(10)
        fig = px.pie(
            values=disease_dist.values,
            names=disease_dist.index,
            title="Top 10 Diseases"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("State Distribution")
        state_dist = filtered_df.groupby('State')['Cases'].sum().sort_values(ascending=False).head(10)
        fig = px.pie(
            values=state_dist.values,
            names=state_dist.index,
            title="Top 10 States"
        )
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 2: TRENDS
# ========================================
with tab2:
    st.header("Temporal Trends Analysis")
    
    # Year-over-year trend
    st.subheader("Year-over-Year Trend")
    yearly_data = filtered_df.groupby('Year').agg({
        'Cases': 'sum',
        'Deaths': 'sum',
        'Unique_ID': 'count'
    }).reset_index()
    yearly_data.columns = ['Year', 'Cases', 'Deaths', 'Reports']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Cases'], name='Cases', marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Deaths'], name='Deaths', marker_color='#d62728'))
    fig.update_layout(
        title='Cases and Deaths by Year',
        xaxis_title='Year',
        yaxis_title='Count',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trend
    st.subheader("Monthly Trend (Aggregated Across Years)")
    monthly_data = filtered_df.groupby('Month_Approx').agg({
        'Cases': 'sum',
        'Deaths': 'sum'
    }).reset_index()
    monthly_data['Month'] = monthly_data['Month_Approx'].map(lambda x: month_names[x-1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_data['Month'], y=monthly_data['Cases'], 
                             mode='lines+markers', name='Cases', line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=monthly_data['Month'], y=monthly_data['Deaths'], 
                             mode='lines+markers', name='Deaths', line=dict(color='#d62728', width=3)))
    fig.update_layout(
        title='Seasonal Pattern (All Years Combined)',
        xaxis_title='Month',
        yaxis_title='Count',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series by disease
    st.subheader("Disease Trend Over Time")
    top_5_diseases = filtered_df.groupby('disease_clean')['Cases'].sum().sort_values(ascending=False).head(5).index
    
    disease_time = filtered_df[filtered_df['disease_clean'].isin(top_5_diseases)].groupby(
        ['Date', 'disease_clean']
    )['Cases'].sum().reset_index()
    
    fig = px.line(
        disease_time,
        x='Date',
        y='Cases',
        color='disease_clean',
        title='Top 5 Diseases - Time Series',
        labels={'disease_clean': 'Disease', 'Cases': 'Weekly Cases'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 3: GEOGRAPHIC
# ========================================
with tab3:
    st.header("Geographic Distribution")
    
    # State-level analysis
    st.subheader("State-Level Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cases by state
        state_data = filtered_df.groupby('State').agg({
            'Cases': 'sum',
            'Deaths': 'sum',
            'Unique_ID': 'count'
        }).reset_index()
        state_data.columns = ['State', 'Cases', 'Deaths', 'Reports']
        state_data['CFR'] = (state_data['Deaths'] / state_data['Cases'] * 100).round(2)
        
        fig = px.bar(
            state_data.sort_values('Cases', ascending=False).head(20),
            x='State',
            y='Cases',
            title='Top 20 States by Cases',
            color='Cases',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Surveillance infrastructure
        fig = px.bar(
            state_data.sort_values('Reports', ascending=False).head(20),
            x='State',
            y='Reports',
            title='Top 20 States by # of Reports (Surveillance Activity)',
            color='Reports',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # District-level analysis (if states selected)
    if selected_states:
        st.subheader("District-Level Analysis")
        
        district_data = filtered_df.groupby(['State', 'District']).agg({
            'Cases': 'sum',
            'Deaths': 'sum'
        }).reset_index()
        
        fig = px.treemap(
            district_data.sort_values('Cases', ascending=False).head(50),
            path=['State', 'District'],
            values='Cases',
            title='Top 50 Districts by Cases (Hierarchical View)',
            color='Cases',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # State comparison table
    st.subheader("State Comparison Table")
    st.dataframe(
        state_data.sort_values('Cases', ascending=False).style.background_gradient(
            subset=['Cases', 'Deaths', 'Reports'], cmap='YlOrRd'
        ),
        use_container_width=True,
        height=400
    )

# ========================================
# TAB 4: DISEASE DEEP DIVE
# ========================================
with tab4:
    st.header("Disease Deep Dive")
    
    # Select disease for analysis
    selected_disease = st.selectbox(
        "Select a disease to analyze",
        options=sorted(filtered_df['disease_clean'].unique())
    )
    
    disease_df = filtered_df[filtered_df['disease_clean'] == selected_disease]
    
    # Disease metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Cases", f"{disease_df['Cases'].sum():,}")
    with col2:
        st.metric("Total Deaths", f"{disease_df['Deaths'].sum():,}")
    with col3:
        cfr = (disease_df['Deaths'].sum() / disease_df['Cases'].sum() * 100) if disease_df['Cases'].sum() > 0 else 0
        st.metric("CFR", f"{cfr:.2f}%")
    with col4:
        st.metric("States Affected", f"{disease_df['State'].nunique()}")
    with col5:
        st.metric("Districts Affected", f"{disease_df['District'].nunique()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top states for this disease
        st.subheader(f"Top States for {selected_disease}")
        disease_states = disease_df.groupby('State')['Cases'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=disease_states.values,
            y=disease_states.index,
            orientation='h',
            labels={'x': 'Cases', 'y': 'State'},
            color=disease_states.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly pattern for this disease
        st.subheader(f"Seasonal Pattern for {selected_disease}")
        disease_monthly = disease_df.groupby('Month_Approx')['Cases'].sum().reset_index()
        disease_monthly['Month'] = disease_monthly['Month_Approx'].map(lambda x: month_names[x-1])
        
        fig = px.line(
            disease_monthly,
            x='Month',
            y='Cases',
            markers=True,
            line_shape='spline'
        )
        fig.update_traces(line=dict(color='#ff7f0e', width=3), marker=dict(size=10))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series for this disease
    st.subheader(f"{selected_disease} - Timeline")
    disease_time = disease_df.groupby('Date')['Cases'].sum().reset_index()
    
    fig = px.area(
        disease_time,
        x='Date',
        y='Cases',
        title=f'{selected_disease} Cases Over Time'
    )
    fig.update_traces(fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4', width=2))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 5: SEASONAL PATTERNS
# ========================================
with tab5:
    st.header("Seasonal Patterns Analysis")
    
    # Heatmap: Disease vs Month
    st.subheader("Disease-Month Heatmap")
    
    # Get top diseases
    top_diseases_seasonal = filtered_df.groupby('disease_clean')['Cases'].sum().sort_values(ascending=False).head(20).index
    
    # Create pivot table
    heatmap_data = filtered_df[filtered_df['disease_clean'].isin(top_diseases_seasonal)].pivot_table(
        values='Cases',
        index='disease_clean',
        columns='Month_Approx',
        aggfunc='sum',
        fill_value=0
    )
    
    # Rename columns to month names
    heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Disease", color="Cases"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Monsoon analysis
    st.subheader("Monsoon Impact Analysis")
    
    # Define monsoon months (June-September)
    filtered_df['Season'] = filtered_df['Month_Approx'].apply(
        lambda x: 'Monsoon' if x in [6, 7, 8, 9] else 'Non-Monsoon'
    )
    
    seasonal_comparison = filtered_df.groupby(['Season', 'disease_clean'])['Cases'].sum().reset_index()
    seasonal_comparison = seasonal_comparison[seasonal_comparison['disease_clean'].isin(top_diseases_seasonal[:10])]
    
    fig = px.bar(
        seasonal_comparison,
        x='disease_clean',
        y='Cases',
        color='Season',
        barmode='group',
        title='Top 10 Diseases: Monsoon vs Non-Monsoon'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Vector-borne diseases seasonal pattern
    st.subheader("Vector-Borne Diseases Seasonal Pattern")
    vector_diseases = ['Dengue', 'Chikungunya', 'Malaria', 'Zika Virus']
    vector_df = filtered_df[filtered_df['disease_clean'].isin(vector_diseases)]
    
    if not vector_df.empty:
        vector_monthly = vector_df.groupby(['Month_Approx', 'disease_clean'])['Cases'].sum().reset_index()
        vector_monthly['Month'] = vector_monthly['Month_Approx'].map(lambda x: month_names[x-1])
        
        fig = px.line(
            vector_monthly,
            x='Month',
            y='Cases',
            color='disease_clean',
            markers=True,
            title='Vector-Borne Diseases - Monthly Pattern'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 6: INSIGHTS
# ========================================
with tab6:
    st.header("Key Insights & Data Quality Assessment")
    
    # Calculate insights from filtered data
    top_disease = filtered_df.groupby('disease_clean')['Cases'].sum().idxmax()
    top_disease_pct = (filtered_df.groupby('disease_clean')['Cases'].sum().max() / filtered_df['Cases'].sum() * 100)
    
    top_state = filtered_df.groupby('State')['Cases'].sum().idxmax()
    top_state_pct = (filtered_df.groupby('State')['Cases'].sum().max() / filtered_df['Cases'].sum() * 100)
    
    # High CFR diseases
    disease_cfr = filtered_df.groupby('disease_clean').agg({
        'Cases': 'sum',
        'Deaths': 'sum'
    })
    disease_cfr['CFR'] = (disease_cfr['Deaths'] / disease_cfr['Cases'] * 100)
    disease_cfr = disease_cfr[disease_cfr['Cases'] >= 10].sort_values('CFR', ascending=False)
    
    # Insight 1: Disease Distribution
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("📊 1. Disease Distribution Bias")
    st.write(f"""
    **Finding:** {top_disease} dominates the dataset with {top_disease_pct:.1f}% of all cases.
    
    **Interpretation:** This likely reflects:
    - Ease of diagnosis (common symptoms)
    - Low barrier to reporting
    - NOT necessarily true disease burden
    
    **Implication:** Surveillance system is biased toward easy-to-diagnose conditions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insight 2: Geographic Bias
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🗺️ 2. Geographic Surveillance Gaps")
    st.write(f"""
    **Finding:** {top_state} accounts for {top_state_pct:.1f}% of all reports.
    
    **States in dataset:** {filtered_df['State'].nunique()} out of 36 states/UTs
    
    **Interpretation:** 
    - Reflects surveillance infrastructure capacity
    - NOT actual disease burden difference
    - Some states severely underrepresented
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insight 3: High-Risk Diseases
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.subheader("⚠️ 3. High-Fatality Diseases (Public Health Priority)")
    st.write("**Top 5 Deadliest Diseases (min 10 cases):**")
    
    for idx, (disease, row) in enumerate(disease_cfr.head(5).iterrows(), 1):
        st.write(f"{idx}. **{disease}**: CFR = {row['CFR']:.2f}% (Cases: {int(row['Cases'])}, Deaths: {int(row['Deaths'])})")
    
    st.write("""
    **Implication:** These diseases should be public health priorities despite lower case counts.
    High CFR indicates need for:
    - Better surveillance
    - Rapid response protocols
    - Public awareness campaigns
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insight 4: Temporal Trends
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("📈 4. Surveillance System Growth")
    
    yearly_growth = filtered_df.groupby('Year')['Unique_ID'].count()
    if len(yearly_growth) > 1:
        growth_rate = ((yearly_growth.iloc[-1] / yearly_growth.iloc[0]) - 1) * 100
        st.write(f"""
        **Finding:** Reports increased from {yearly_growth.iloc[0]} (2020) to {yearly_growth.iloc[-1]} ({yearly_growth.index[-1]})
        
        **Growth:** {growth_rate:.1f}% over {len(yearly_growth)} years
        
        **Interpretation:** This likely reflects:
        - Improved surveillance infrastructure
        - Better reporting compliance
        - NOT necessarily increase in disease burden
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insight 5: Data Quality
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🔍 5. Data Quality Assessment")
    
    original_diseases = 216  # From your EDA
    cleaned_diseases = df['disease_clean'].nunique()
    reduction = ((original_diseases - cleaned_diseases) / original_diseases * 100)
    
    st.write(f"""
    **Original disease labels:** {original_diseases}
    **After standardization:** {cleaned_diseases}
    **Complexity reduction:** {reduction:.1f}%
    
    **Key Quality Issues Found:**
    - Spelling variants (e.g., "chickenpox" vs "chickenpo x")
    - Syntax differences (e.g., "acute diarrheal disease" vs "diarrheal disease acute")
    - Location data mixed with disease names
    - Completely malformed entries
    
    **Solution Implemented:**
    - Custom disease ontology mapping
    - Standardized naming convention
    - Preserved epidemiological validity
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations for Surveillance System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Quality Improvements:**
        1. ✅ Implement standardized disease dropdown (not free text)
        2. ✅ Add input validation at data entry
        3. ✅ Create national disease ontology
        4. ✅ Regular data quality audits
        """)
    
    with col2:
        st.markdown("""
        **Coverage Improvements:**
        1. ✅ Strengthen surveillance in underrepresented states
        2. ✅ Incentivize reporting from remote districts
        3. ✅ Provide training on disease identification
        4. ✅ Integrate with hospital management systems
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>India Disease Surveillance Dashboard</strong></p>
    <p>Data: 2020-2025 | Built with Streamlit & Plotly</p>
    <p>⚠️ This dashboard is for analytical purposes. Consult official health authorities for medical decisions.</p>
</div>
""", unsafe_allow_html=True)
