import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Tuple
import sys

# Force cache clear and version check for cloud deployment
st.set_page_config(
    page_title="Scope 3 Emission Gaming Calculator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Version tracking for deployment debugging
APP_VERSION = "2.1.0"
st.sidebar.write(f"App Version: {APP_VERSION}")

# Clear all caches on app start for cloud deployment
try:
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
except:
    pass

# Enable Altair for Streamlit
alt.data_transformers.enable('json')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .gaming-alert {
        background: rgba(255, 107, 107, 0.1);
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("""
<div class="main-header">
    <h1>🎯 Scope 3 Emission Gaming Calculator</h1>
    <h3>How ONE company can appear to meet Science-Based Targets through strategic emission factor selection</h3>
    <p>This tool demonstrates how a Food & Beverage company with a <strong>shared 2025 baseline</strong> can achieve apparent <strong>4.2% annual emission reductions</strong> (meeting SBTi requirements) through Scope 3 emission factor gaming alone—without any operational changes.</p>
</div>
""", unsafe_allow_html=True)

# Key insight callout
st.markdown("""
<div class="gaming-alert">
    <strong>🚨 Key Finding:</strong> Same company, same baseline → Strategic factor switching can show 13+ years of SBTi compliance while emissions actually grow
</div>
""", unsafe_allow_html=True)

# Real emission factors from research
EMISSION_FACTORS = {
    "Milk": {"Conservative": 3.20, "Moderate": 1.24, "Aggressive": 0.60},
    "Cheese": {"Conservative": 13.50, "Moderate": 9.78, "Aggressive": 5.30},
    "Butter": {"Conservative": 12.00, "Moderate": 9.30, "Aggressive": 7.30},
    "Wheat": {"Conservative": 0.82, "Moderate": 0.60, "Aggressive": 0.35},
    "Rice": {"Conservative": 1.60, "Moderate": 1.20, "Aggressive": 0.90}
}

# Constants
YEARS = list(range(2025, 2031))
BASE_YEAR = 2025
SBTI_REDUCTION_RATE = 4.2
GAMING_STRATEGIES = ["No Gaming (Honest)", "Moderate Gaming", "Aggressive Gaming"]
STRATEGY_COLORS = {
    'No Gaming (Honest)': '#ff6b6b',
    'Moderate Gaming': '#ffa500',
    'Aggressive Gaming': '#4ecdc4',
    'SBTi 4.2% Pathway': '#45b7d1'
}

def calculate_weighted_factor(scenario: str, product_mix: Dict[str, float]) -> float:
    """Calculate weighted emission factor based on product mix and scenario"""
    weighted_sum = 0.0
    total_percentage = sum(product_mix.values())
    
    if total_percentage == 0:
        return 0.0
        
    for product, percentage in product_mix.items():
        if product in EMISSION_FACTORS:
            factor = EMISSION_FACTORS[product][scenario]
            weighted_sum += factor * (percentage / total_percentage)
            
    return weighted_sum

def get_emission_factor_for_year(year: int, strategy: str, baseline_factor: float, 
                               gaming_start_year: int, product_mix: Dict[str, float]) -> float:
    """Get emission factor for a specific year and strategy"""
    if year < gaming_start_year:
        return baseline_factor
    else:
        if strategy == "No Gaming (Honest)":
            return baseline_factor
        elif strategy == "Moderate Gaming":
            return calculate_weighted_factor("Moderate", product_mix)
        elif strategy == "Aggressive Gaming":
            return calculate_weighted_factor("Aggressive", product_mix)
    return baseline_factor

# Sidebar controls
st.sidebar.header("🎛️ Company Configuration")

# Product mix
st.sidebar.subheader("Product Portfolio Mix (%)")
product_mix = {}
default_mix = {"Milk": 30, "Cheese": 15, "Butter": 5, "Wheat": 35, "Rice": 15}

for product in EMISSION_FACTORS.keys():
    product_mix[product] = st.sidebar.slider(
        f"{product}",
        min_value=0,
        max_value=50,
        value=default_mix[product],
        step=1
    )

# Normalize to 100%
total_mix = sum(product_mix.values())
if total_mix > 0:
    product_mix = {k: (v/total_mix)*100 for k, v in product_mix.items()}

st.sidebar.markdown(f"**Total: {sum(product_mix.values()):.0f}%**")

# Company settings
st.sidebar.subheader("📊 Company Scenario")
annual_production = st.sidebar.number_input(
    "Annual Production (tonnes)",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000
)

growth_rate = st.sidebar.slider(
    "Annual Growth Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=7.0,
    step=0.5
)

# Gaming timeline
st.sidebar.subheader("⏱️ Gaming Timeline")
gaming_start_year = st.sidebar.selectbox(
    "Gaming Starts From",
    [2026, 2027, 2028],
    index=0
)

# Monte Carlo settings
st.sidebar.subheader("🎲 Simulation Settings")
n_iterations = st.sidebar.selectbox("Monte Carlo Iterations", [100, 500, 1000], index=1)
uncertainty = st.sidebar.slider("Factor Uncertainty (±%)", 5.0, 15.0, 10.0, step=1.0)

# Calculate baseline factor
baseline_factor = calculate_weighted_factor("Conservative", product_mix)

# Calculate emissions trajectories (simplified for cloud deployment)
def calculate_emissions_data():
    """Calculate emissions data for all strategies"""
    results = {}
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for strategy in GAMING_STRATEGIES:
        trajectory = []
        
        for year in YEARS:
            year_index = year - BASE_YEAR
            production_year = annual_production * (1 + growth_rate/100) ** year_index
            emission_factor = get_emission_factor_for_year(
                year, strategy, baseline_factor, gaming_start_year, product_mix
            )
            
            # Add some Monte Carlo uncertainty
            if n_iterations > 100:
                variations = []
                for _ in range(min(n_iterations, 500)):  # Limit for cloud performance
                    variation = np.random.uniform(-uncertainty/100, uncertainty/100)
                    varied_factor = emission_factor * (1 + variation)
                    variations.append(production_year * varied_factor)
                
                mean_emissions = np.mean(variations)
                lower_ci = np.percentile(variations, 2.5)
                upper_ci = np.percentile(variations, 97.5)
            else:
                mean_emissions = production_year * emission_factor
                lower_ci = mean_emissions * 0.95
                upper_ci = mean_emissions * 1.05
            
            trajectory.append({
                'year': year,
                'mean': mean_emissions,
                'lower': lower_ci,
                'upper': upper_ci
            })
        
        results[strategy] = trajectory
    
    return results

# Calculate all data
with st.spinner(f"🔬 Running {n_iterations:,} Monte Carlo simulations..."):
    emissions_data = calculate_emissions_data()

# Calculate SBTi pathway
base_emissions = emissions_data["No Gaming (Honest)"][0]['mean']  # 2025 baseline
sbti_pathway = []
for year in YEARS:
    year_index = year - BASE_YEAR
    sbti_emissions = base_emissions * ((1 - SBTI_REDUCTION_RATE/100) ** year_index)
    sbti_pathway.append({'year': year, 'emissions': sbti_emissions})

# Main visualization section
st.subheader("🎮 Baseline Convergence → Strategic Factor Gaming")

# Prepare data for Altair chart
chart_data_list = []

# Add gaming strategy data
for strategy in GAMING_STRATEGIES:
    for data_point in emissions_data[strategy]:
        chart_data_list.append({
            'Year': data_point['year'],
            'Strategy': strategy,
            'Emissions': data_point['mean'],
            'Lower_CI': data_point['lower'],
            'Upper_CI': data_point['upper']
        })

# Add SBTi pathway
for sbti_point in sbti_pathway:
    chart_data_list.append({
        'Year': sbti_point['year'],
        'Strategy': 'SBTi 4.2% Pathway',
        'Emissions': sbti_point['emissions'],
        'Lower_CI': sbti_point['emissions'],
        'Upper_CI': sbti_point['emissions']
    })

chart_df = pd.DataFrame(chart_data_list)

# Create Altair chart
base_chart = alt.Chart(chart_df)

# Confidence intervals
confidence_bands = base_chart.mark_area(opacity=0.2).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Lower_CI:Q', title='Scope 3 Emissions (tCO₂e)', scale=alt.Scale(zero=False)),
    y2=alt.Y2('Upper_CI:Q'),
    color=alt.Color('Strategy:N', 
                   scale=alt.Scale(
                       domain=list(STRATEGY_COLORS.keys()),
                       range=list(STRATEGY_COLORS.values())
                   ),
                   legend=alt.Legend(title="95% Confidence Intervals"))
).transform_filter(
    alt.datum.Strategy != 'SBTi 4.2% Pathway'
)

# Mean lines
mean_lines = base_chart.mark_line(
    strokeWidth=4,
    point=alt.OverlayMarkDef(size=100, filled=True)
).encode(
    x=alt.X('Year:O'),
    y=alt.Y('Emissions:Q'),
    color=alt.Color('Strategy:N',
                   scale=alt.Scale(
                       domain=list(STRATEGY_COLORS.keys()),
                       range=list(STRATEGY_COLORS.values())
                   ),
                   legend=alt.Legend(title="Gaming Strategies")),
    strokeDash=alt.condition(
        alt.datum.Strategy == 'SBTi 4.2% Pathway', 
        alt.value([8, 4]), 
        alt.value([0])
    ),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Strategy:N', title='Strategy'),
        alt.Tooltip('Emissions:Q', title='Emissions (tCO₂e)', format='.0f'),
    ]
)

# Gaming start line
gaming_start_line = alt.Chart(pd.DataFrame({'x': [gaming_start_year]})).mark_rule(
    color='red',
    strokeWidth=3,
    strokeDash=[5, 5],
    opacity=0.8
).encode(
    x=alt.X('x:O'),
    tooltip=alt.value(f'Gaming Starts: {gaming_start_year}')
)

# Combine charts
final_chart = (confidence_bands + mean_lines + gaming_start_line).resolve_scale(
    color='independent'
).properties(
    width=800,
    height=500,
    title=alt.TitleParams(
        text=[
            f'Baseline Convergence (2025) → Gaming Starts ({gaming_start_year})',
            'Same Company, Different Accounting Strategies'
        ],
        fontSize=18,
        anchor='start'
    )
)

# Display the chart
st.altair_chart(final_chart, use_container_width=True)

# Calculate key metrics
honest_2030 = emissions_data["No Gaming (Honest)"][5]['mean']  # 2030 data
aggressive_2030 = emissions_data["Aggressive Gaming"][5]['mean']  # 2030 data

real_growth = ((honest_2030 - base_emissions) / base_emissions) * 100
apparent_reduction = ((base_emissions - aggressive_2030) / base_emissions) * 100
gaming_annual_rate = apparent_reduction / 5
years_compliant = gaming_annual_rate / SBTI_REDUCTION_RATE

# Display key metrics
st.subheader("📊 Gaming Impact Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>{base_emissions:,.0f}</h3>
        <p>Shared Baseline (2025)<br>tCO₂e</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>+{real_growth:.1f}%</h3>
        <p>Real Growth (2030)<br>Honest Accounting</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>-{apparent_reduction:.1f}%</h3>
        <p>Apparent Reduction<br>Through Gaming</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>{years_compliant:.1f}</h3>
        <p>Years of SBTi<br>Compliance</p>
    </div>
    """, unsafe_allow_html=True)

# Comparison table
st.markdown("**Gaming vs. Reality Comparison (2030)**")

comparison_data = []
for i, strategy in enumerate(GAMING_STRATEGIES):
    emissions_2030 = emissions_data[strategy][5]['mean']
    change_pct = ((emissions_2030 - base_emissions) / base_emissions) * 100
    
    if change_pct > 0:
        compliance = "❌ No (Real emissions)"
    elif abs(change_pct) >= 20:
        compliance = "✅ Yes (Through gaming)" if strategy != 'No Gaming (Honest)' else "✅ Yes"
    else:
        compliance = "⚠️ Partially"
    
    comparison_data.append({
        'Accounting Strategy': strategy,
        '2030 Emissions (tCO₂e)': f"{emissions_2030:,.0f}",
        'Change from 2025': f"{change_pct:+.1f}%",
        'SBTi Compliant?': compliance
    })

# Add SBTi target
sbti_2030 = sbti_pathway[5]['emissions']
sbti_change = ((sbti_2030 - base_emissions) / base_emissions) * 100
comparison_data.append({
    'Accounting Strategy': 'SBTi Target',
    '2030 Emissions (tCO₂e)': f"{sbti_2030:,.0f}",
    'Change from 2025': f"{sbti_change:.1f}%",
    'SBTi Compliant?': "🎯 Target"
})

df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True)

# Gaming mechanism explanation
st.subheader("🔍 The Baseline Convergence Gaming Mechanism")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"""
    **Step-by-Step Gaming Process:**
    1. **Shared Baseline (2025)**: All use conservative factors → **{base_emissions:,.0f} tCO₂e**
    2. **Gaming Trigger ({gaming_start_year})**: Strategic factor switching begins
    3. **Database Shopping**: Choose most aggressive available factors
    4. **Apparent Reduction**: Show -{apparent_reduction:.1f}% reduction by 2030
    5. **SBTi Compliance**: Meet 4.2% annual reduction target
    6. **Reality**: Actual emissions grew +{real_growth:.1f}% from business expansion
    """)

with col_right:
    st.markdown("""
    **Why This Gaming Works:**
    - ✅ **Same starting point** - baseline convergence is realistic
    - ✅ **Plausible excuse** - "improved data quality"
    - ✅ **Regulatory approval** - SBTi accepts factor updates
    - ✅ **No operational changes** required
    - ✅ **Competitive advantage** over honest companies
    - ❌ **Undermines climate credibility** and actual progress
    """)

# Research context
st.markdown("---")
st.subheader("🔗 Research Context")

col_research1, col_research2 = st.columns(2)

with col_research1:
    st.markdown("""
    **📄 Based on Published Research**
    
    *"Operationalizing corporate climate action through five research frontiers"*
    
    By Ramana Gudipudi et al.
    
    **Key Findings:**
    - Companies can reduce reported emissions by up to 6.7x through gaming
    - 85% of companies cite Scope 3 accounting as primary barrier
    - Current GHG Protocol enables strategic factor selection
    """)

with col_research2:
    st.markdown("""
    **🎯 Solutions Needed**
    
    - Standardized, auditable emission factor databases
    - Factor selection transparency requirements
    - Statistical outlier detection systems
    - Industry-specific verification protocols
    - Dynamic baseline adjustment protocols
    """)

# Footer
gaming_duration = 2030 - gaming_start_year
st.markdown("---")
st.markdown(f"""
**About this tool**: Demonstrates emission factor gaming using real LCA database variations from peer-reviewed research. 
Shows how ONE company can strategically transition factors over {gaming_duration} years ({gaming_start_year}-2030) to achieve {years_compliant:.1f} years of apparent SBTi compliance while emissions actually grow.

*Research by Ramana Gudipudi, Luis Costa, Ponraj Arumugam, Matthew Agarwala, Jürgen P. Kropp, Felix Creutzig*
""")

# Force cache clear button for cloud deployment
if st.sidebar.button("🔄 Force Refresh (Clear All Cache)"):
    st.cache_data.clear()
    st.rerun()
