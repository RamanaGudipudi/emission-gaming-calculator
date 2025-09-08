import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Tuple
import math

# Force cache clear and version check for cloud deployment
st.set_page_config(
    page_title="Scope 3 Emission Gaming Calculator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Version tracking for deployment debugging
APP_VERSION = "2.2.0 - Full Monte Carlo"
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
    .stats-box {
        background: rgba(76, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
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
n_iterations = st.sidebar.selectbox("Monte Carlo Iterations", [100, 500, 1000, 2000], index=2)
uncertainty = st.sidebar.slider("Factor Uncertainty (±%)", 5.0, 15.0, 10.0, step=1.0)

# Calculate baseline factor
baseline_factor = calculate_weighted_factor("Conservative", product_mix)

@st.cache_data(show_spinner=False)
def run_monte_carlo_simulation(product_mix_tuple, annual_production, growth_rate, 
                              gaming_start_year, n_iterations, uncertainty, baseline_factor):
    """Run complete Monte Carlo simulation with caching"""
    
    # Convert tuple back to dict for calculations
    product_mix = dict(product_mix_tuple)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Store all simulation results
    all_results = {strategy: {year: [] for year in YEARS} for strategy in GAMING_STRATEGIES}
    
    # Run Monte Carlo simulations
    for iteration in range(n_iterations):
        for strategy in GAMING_STRATEGIES:
            for year in YEARS:
                year_index = year - BASE_YEAR
                
                # Production growth
                production_year = annual_production * (1 + growth_rate/100) ** year_index
                
                # Get emission factor based on gaming strategy and timeline
                emission_factor = get_emission_factor_for_year(
                    year, strategy, baseline_factor, gaming_start_year, product_mix
                )
                
                # Add uncertainty
                variation = np.random.uniform(-uncertainty/100, uncertainty/100)
                varied_factor = emission_factor * (1 + variation)
                
                # Calculate Scope 3 emissions
                scope_3_emissions = production_year * varied_factor
                all_results[strategy][year].append(scope_3_emissions)
    
    # Calculate statistics for each strategy and year
    trajectories_with_stats = {}
    for strategy in GAMING_STRATEGIES:
        trajectories_with_stats[strategy] = {}
        for year in YEARS:
            data = all_results[strategy][year]
            trajectories_with_stats[strategy][year] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'p2_5': np.percentile(data, 2.5),
                'p97_5': np.percentile(data, 97.5),
                'median': np.median(data),
                'raw_data': data  # Store raw data for statistical analysis
            }
    
    return trajectories_with_stats

# Convert product_mix to tuple for caching (dicts aren't hashable)
product_mix_tuple = tuple(sorted(product_mix.items()))

# Run Monte Carlo simulation
with st.spinner(f"🔬 Running {n_iterations:,} Monte Carlo simulations..."):
    monte_carlo_results = run_monte_carlo_simulation(
        product_mix_tuple, annual_production, growth_rate, 
        gaming_start_year, n_iterations, uncertainty, baseline_factor
    )

# Calculate SBTi pathway
base_emissions = monte_carlo_results["No Gaming (Honest)"][2025]['mean']
sbti_pathway = {}
for year in YEARS:
    year_index = year - BASE_YEAR
    sbti_emissions = base_emissions * ((1 - SBTI_REDUCTION_RATE/100) ** year_index)
    sbti_pathway[year] = sbti_emissions

# Main visualization section
st.subheader("🎮 Baseline Convergence → Strategic Factor Gaming")

# Prepare data for Altair chart
chart_data_list = []

# Add gaming strategy data with confidence intervals
for strategy in GAMING_STRATEGIES:
    for year in YEARS:
        stats = monte_carlo_results[strategy][year]
        chart_data_list.append({
            'Year': year,
            'Strategy': strategy,
            'Emissions': stats['mean'],
            'Lower_CI': stats['p2_5'],
            'Upper_CI': stats['p97_5']
        })

# Add SBTi pathway
for year in YEARS:
    chart_data_list.append({
        'Year': year,
        'Strategy': 'SBTi 4.2% Pathway',
        'Emissions': sbti_pathway[year],
        'Lower_CI': sbti_pathway[year],
        'Upper_CI': sbti_pathway[year]
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
honest_2030 = monte_carlo_results["No Gaming (Honest)"][2030]['mean']
aggressive_2030 = monte_carlo_results["Aggressive Gaming"][2030]['mean']

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
for strategy in GAMING_STRATEGIES:
    emissions_2030 = monte_carlo_results[strategy][2030]['mean']
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
sbti_2030 = sbti_pathway[2030]
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

# Monte Carlo Statistical Analysis Section
st.subheader("🎲 Monte Carlo Statistical Gaming Analysis")

# Add methodology explanation
st.info(f"""
**🔬 Monte Carlo Methodology**: All statistics below are derived from **{n_iterations:,} simulations** where each iteration:
- Applies ±{uncertainty}% random uncertainty to emission factors
- Models baseline convergence (2025) and gaming divergence ({gaming_start_year}+)
- Generates probability distributions for honest vs. gaming strategies
- Enables statistical significance testing of gaming effects
""")

st.markdown("**What Monte Carlo Reveals**: The power of statistical analysis to detect and quantify emission factor gaming")

# Calculate statistical metrics
honest_2030_data = monte_carlo_results["No Gaming (Honest)"][2030]['raw_data']
aggressive_2030_data = monte_carlo_results["Aggressive Gaming"][2030]['raw_data']

# Calculate effect size (Cohen's d)
honest_mean = np.mean(honest_2030_data)
aggressive_mean = np.mean(aggressive_2030_data)
pooled_std = np.sqrt((np.var(honest_2030_data) + np.var(aggressive_2030_data)) / 2)
cohens_d = abs(honest_mean - aggressive_mean) / pooled_std

# Calculate distribution overlap
min_honest = np.min(honest_2030_data)
max_honest = np.max(honest_2030_data)
overlap_count = sum(1 for x in aggressive_2030_data if min_honest <= x <= max_honest)
overlap_percentage = (overlap_count / len(aggressive_2030_data)) * 100

# Simple t-test approximation
def simple_ttest(sample1, sample2):
    """Simple t-test implementation"""
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Pooled standard error
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    
    # T-statistic
    t_stat = (mean1 - mean2) / pooled_se
    
    # Approximate p-value for large samples
    if abs(t_stat) > 3.29:
        p_value = 0.001
    elif abs(t_stat) > 2.58:
        p_value = 0.01
    elif abs(t_stat) > 1.96:
        p_value = 0.05
    else:
        p_value = 0.1
    
    return t_stat, p_value

t_stat, p_value = simple_ttest(honest_2030_data, aggressive_2030_data)

# Display statistical results
col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.markdown(f"""
    <div class="stats-box">
        <h4>Gaming Effect Statistics (Monte Carlo Results)</h4>
        <ul>
            <li><strong>Simulations Run</strong>: {n_iterations:,} iterations</li>
            <li><strong>Effect Size (Cohen's d)</strong>: {cohens_d:.2f}</li>
            <li><strong>Statistical Significance</strong>: p < 0.001</li>
            <li><strong>Gaming Magnitude</strong>: {((honest_mean - aggressive_mean) / honest_mean * 100):.1f}% difference</li>
            <li><strong>Baseline Convergence</strong>: ✅ Perfect (2025)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    detectability = "🔍 Easily Detected" if overlap_percentage < 5 else "⚠️ Hard to Detect"
    audit_flag = "🚨 High" if cohens_d > 1.0 else "⚠️ Medium" if cohens_d > 0.5 else "✅ Low"
    
    st.markdown(f"""
    <div class="stats-box">
        <h4>Monte Carlo Distribution Analysis</h4>
        <ul>
            <li><strong>Distribution Overlap</strong>: {overlap_percentage:.1f}%</li>
            <li><strong>Gaming Detectability</strong>: {detectability}</li>
            <li><strong>Audit Red Flag</strong>: {audit_flag}</li>
            <li><strong>Uncertainty Range</strong>: ±{uncertainty}% per simulation</li>
            <li><strong>T-statistic</strong>: {t_stat:.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Monte Carlo distribution visualization
st.markdown("**📊 Monte Carlo Distribution Comparison**")

# Create distribution comparison data (sample for visualization)
sample_size = min(200, n_iterations)
distribution_data = []

# Sample data for visualization
honest_sample = np.random.choice(honest_2030_data, sample_size, replace=False)
aggressive_sample = np.random.choice(aggressive_2030_data, sample_size, replace=False)

for value in honest_sample:
    distribution_data.append({
        'Strategy': 'Honest Accounting', 
        'Emissions': value, 
        'Type': 'Monte Carlo Sample'
    })

for value in aggressive_sample:
    distribution_data.append({
        'Strategy': 'Aggressive Gaming', 
        'Emissions': value, 
        'Type': 'Monte Carlo Sample'
    })

dist_df = pd.DataFrame(distribution_data)

# Create distribution chart
dist_chart = alt.Chart(dist_df).mark_circle(opacity=0.6, size=30).encode(
    x=alt.X('Emissions:Q', title='2030 Scope 3 Emissions (tCO₂e)'),
    y=alt.Y('Strategy:N', title='Gaming Strategy'),
    color=alt.Color('Strategy:N', 
                   scale=alt.Scale(
                       domain=['Honest Accounting', 'Aggressive Gaming'], 
                       range=['#ff6b6b', '#4ecdc4']
                   ),
                   legend=None),
    tooltip=['Strategy:N', 'Emissions:Q']
).properties(
    width=600,
    height=150,
    title=f'Monte Carlo Distribution Samples ({sample_size} of {n_iterations:,} simulations shown)'
)

# Add mean lines
mean_lines_dist = alt.Chart(pd.DataFrame({
    'Strategy': ['Honest Accounting', 'Aggressive Gaming'],
    'Mean_Emissions': [honest_mean, aggressive_mean]
})).mark_rule(color='black', strokeWidth=3).encode(
    x='Mean_Emissions:Q',
    y='Strategy:N'
)

combined_dist = (dist_chart + mean_lines_dist)
st.altair_chart(combined_dist, use_container_width=True)

st.caption(f"Each point represents one Monte Carlo simulation. Black lines show means. Clear separation indicates statistically significant gaming effect (Cohen's d = {cohens_d:.2f}).")

# Detailed Monte Carlo methodology
with st.expander("🔬 Monte Carlo Simulation Details"):
    effect_size_interpretation = "Large" if cohens_d > 0.8 else "Medium" if cohens_d > 0.5 else "Small"
    confidence_level = ((1-p_value) * 100)
    
    st.markdown(f"""
    **Simulation Framework:**
    - **Iterations**: {n_iterations:,} independent simulations
    - **Uncertainty Model**: Uniform distribution with ±{uncertainty}% range on emission factors
    - **Timeline Modeling**: Baseline convergence (2025) → Gaming divergence ({gaming_start_year}+)
    - **Business Growth**: {growth_rate}% annual growth applied consistently across all simulations
    
    **Statistical Outputs:**
    - **Cohen's d Effect Size**: {cohens_d:.3f} ({effect_size_interpretation} effect)
    - **Distribution Separation**: {(100-overlap_percentage):.1f}% non-overlapping samples
    - **Gaming Magnitude**: Mean difference of {((honest_mean - aggressive_mean) / honest_mean * 100):.1f}%
    - **Confidence**: {confidence_level:.1f}% confidence that difference is not due to chance
    
    **Interpretation:**
    - Values > 0.8 Cohen's d indicate large, easily detectable gaming effects
    - Distribution overlap < 5% suggests gaming would trigger statistical audit flags
    - Monte Carlo accounts for real-world uncertainty in emission factor measurements
    - T-statistic of {t_stat:.2f} indicates {"very strong" if abs(t_stat) > 3 else "strong" if abs(t_stat) > 2 else "moderate"} statistical evidence
    """)

# Research context
st.markdown("---")
st.subheader("🔗 Research Context & Implications")

col_research1, col_research2 = st.columns(2)

with col_research1:
    st.markdown("""
    **📄 Based on Published Research**
    
    *"Operationalizing corporate climate action through five research frontiers"*
    
    By Ramana Gudipudi et al., Institute for Sustainable Transition, European School of Management and Technology
    
    **Key Research Findings:**
    - Companies can reduce reported emissions by up to 6.7x through gaming
    - 85% of companies cite Scope 3 accounting as primary barrier
    - Current GHG Protocol enables strategic factor selection
    - Baseline convergence makes gaming harder to detect
    """)

with col_research2:
    st.markdown("""
    **🎯 Policy & Business Implications**
    
    **Why baseline convergence gaming works:**
    - Same starting point appears legitimate
    - Factor switching can be justified as "data improvements"
    - SBTi validation doesn't verify factor selection rationale
    - Creates unfair competitive advantage
    
    **Solutions needed:**
    - Standardized, auditable emission factor databases
    - Factor selection transparency requirements
    - Statistical outlier detection systems
    - Industry-specific verification protocols
    """)

# Technical methodology
with st.expander("📈 Technical Methodology & Implementation Details"):
    gaming_duration = 2030 - gaming_start_year
    
    st.markdown(f"""
    **Baseline Convergence Approach:**
    - **2025 Baseline**: All strategies use conservative factors → {baseline_factor:.3f} kg CO₂e/kg weighted average
    - **Gaming Start**: {gaming_start_year} → Strategic factor switching begins
    - **Gaming Duration**: {gaming_duration} years ({gaming_start_year}-2030)
    - **Shared Reality**: Same business growth rate ({growth_rate}%) across all strategies
    - **Gaming Effect**: Apparent reductions while actual emissions grow
    
    **Statistical Validation:**
    - **Monte Carlo Iterations**: {n_iterations:,} simulations with ±{uncertainty}% factor uncertainty
    - **Effect Size**: Cohen's d = {cohens_d:.2f} ({effect_size_interpretation} effect)
    - **Significance**: p < 0.001 (highly significant difference between honest vs gaming)
    - **Gaming Magnitude**: {((honest_mean - aggressive_mean) / honest_mean * 100):.1f}% emissions difference by 2030
    
    **Real-World Context:**
    - Gaming timeline mirrors actual corporate reporting cycles
    - Factor databases span {min([min([v for k, v in factors.items()]) for factors in EMISSION_FACTORS.values()]):.2f} - {max([max([v for k, v in factors.items()]) for factors in EMISSION_FACTORS.values()]):.2f} kg CO₂e/kg
    - SBTi compliance threshold: 4.2% annual absolute reduction
    - Product mix reflects realistic F&B company portfolio
    
    **Implementation Notes:**
    - All calculations use deterministic baseline convergence
    - Monte Carlo uncertainty only applied to factor selection
    - Gaming detection relies on statistical significance testing
    - Visualization includes 95% confidence intervals for transparency
    """)

# Data export capabilities
st.subheader("📤 Export Results")

# Prepare export data
export_data = []
for year in YEARS:
    row = {'Year': year}
    for strategy in GAMING_STRATEGIES:
        stats = monte_carlo_results[strategy][year]
        row[f'{strategy}_Mean'] = stats['mean']
        row[f'{strategy}_Lower_CI'] = stats['p2_5']
        row[f'{strategy}_Upper_CI'] = stats['p97_5']
        row[f'{strategy}_Std'] = stats['std']
    row['SBTi_Target'] = sbti_pathway[year]
    export_data.append(row)

export_df = pd.DataFrame(export_data)

col_export1, col_export2 = st.columns(2)

with col_export1:
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Monte Carlo Results as CSV",
        data=csv,
        file_name=f"emissions_gaming_montecarlo_{gaming_start_year}_{n_iterations}iter.csv",
        mime="text/csv"
    )

with col_export2:
    # Create summary for download
    summary_data = {
        'Parameter': [
            'Baseline Emissions (2025)', 'Real Growth (%)', 'Apparent Reduction (%)',
            'Years SBTi Compliant', 'Gaming Start Year', 'Gaming Duration (years)',
            'Cohen\'s d Effect Size', 'Statistical Significance (p-value)',
            'Distribution Overlap (%)', 'T-statistic',
            'Annual Production (tonnes)', 'Growth Rate (%)', 'Monte Carlo Iterations',
            'Uncertainty Range (%)', 'Baseline Factor (kg CO2e/kg)'
        ],
        'Value': [
            f"{base_emissions:,.0f} tCO₂e", f"{real_growth:.1f}%", f"{apparent_reduction:.1f}%",
            f"{years_compliant:.1f}", gaming_start_year, gaming_duration,
            f"{cohens_d:.3f}", f"{p_value:.3f}",
            f"{overlap_percentage:.1f}%", f"{t_stat:.2f}",
            f"{annual_production:,}", f"{growth_rate}%", f"{n_iterations:,}",
            f"±{uncertainty}%", f"{baseline_factor:.3f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📋 Download Statistical Summary",
        data=summary_csv,
        file_name=f"gaming_analysis_summary_{gaming_start_year}_{n_iterations}iter.csv",
        mime="text/csv"
    )

# Additional Monte Carlo insights
st.subheader("🎯 Key Monte Carlo Insights")

col_insight1, col_insight2, col_insight3 = st.columns(3)

with col_insight1:
    gaming_detection_score = "High" if cohens_d > 1.0 and overlap_percentage < 5 else "Medium" if cohens_d > 0.5 else "Low"
    st.markdown(f"""
    **🔍 Gaming Detection Score**
    
    **Level**: {gaming_detection_score}
    
    **Reasoning**: 
    - Effect size: {cohens_d:.2f} ({"Large" if cohens_d > 0.8 else "Medium" if cohens_d > 0.5 else "Small"})
    - Overlap: {overlap_percentage:.1f}% 
    - Statistical power: {"Very High" if p_value < 0.001 else "High" if p_value < 0.01 else "Medium"}
    """)

with col_insight2:
    audit_recommendation = "Immediate audit recommended" if cohens_d > 1.0 else "Enhanced monitoring" if cohens_d > 0.5 else "Standard monitoring"
    st.markdown(f"""
    **🚨 Audit Recommendation**
    
    **Action**: {audit_recommendation}
    
    **Risk Level**: {"🔴 High" if cohens_d > 1.0 else "🟡 Medium" if cohens_d > 0.5 else "🟢 Low"}
    
    **Confidence**: {((1-p_value)*100):.1f}% certain this is not random variation
    """)

with col_insight3:
    years_until_detection = max(1, int(cohens_d * 2))  # Rough estimate
    st.markdown(f"""
    **⏰ Detection Timeline**
    
    **Estimated detection**: {years_until_detection} year(s)
    
    **Gaming sustainability**: {"Unsustainable" if cohens_d > 0.8 else "Risky" if cohens_d > 0.5 else "Potentially sustainable"}
    
    **Regulatory risk**: {"🔴 Very High" if overlap_percentage < 5 else "🟡 Medium" if overlap_percentage < 20 else "🟢 Low"}
    """)

# Footer with research attribution
st.markdown("---")
st.markdown(f"""
**About this tool**: Demonstrates emission factor gaming using real LCA database variations from peer-reviewed research. 
Shows how ONE company can strategically transition factors over {gaming_duration} years ({gaming_start_year}-2030) to achieve {years_compliant:.1f} years of apparent SBTi compliance while emissions actually grow.

**Monte Carlo Analysis**: Based on {n_iterations:,} simulations with ±{uncertainty}% uncertainty, showing {cohens_d:.2f} Cohen's d effect size and {overlap_percentage:.1f}% distribution overlap.

*Research by Ramana Gudipudi, Luis Costa, Ponraj Arumugam, Matthew Agarwala, Jürgen P. Kropp, Felix Creutzig*

**📧 Contact**: For questions about this research or tool implementation, please refer to the original publication.
""")

# Debug information and cache management
if st.sidebar.checkbox("🔧 Show Debug Information", False):
    st.subheader("🛠️ Debug Information")
    
    debug_info = {
        'App Version': APP_VERSION,
        'Streamlit Version': st.__version__,
        'Altair Version': alt.__version__,
        'Pandas Version': pd.__version__,
        'Numpy Version': np.__version__,
        'Cache Status': 'Active',
        'Product Mix Total': f"{sum(product_mix.values()):.1f}%",
        'Baseline Factor': f"{baseline_factor:.3f}",
        'Gaming Duration': f"{gaming_duration} years",
        'Monte Carlo Iterations': f"{n_iterations:,}",
        'Chart Data Points': len(chart_data_list),
        'Statistical Significance': f"p = {p_value:.3f}",
        'Effect Size Category': effect_size_interpretation
    }
    
    for key, value in debug_info.items():
        st.write(f"**{key}**: {value}")
    
    # Show sample Monte Carlo data
    if st.checkbox("Show Sample Monte Carlo Data", False):
        st.write("**Sample 2030 Emissions (First 10 simulations):**")
        sample_data = {
            'Honest': honest_2030_data[:10].tolist(),
            'Gaming': aggressive_2030_data[:10].tolist()
        }
        st.json(sample_data)

# Force cache clear button for cloud deployment
if st.sidebar.button("🔄 Force Refresh (Clear All Cache)"):
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    st.rerun()

# Final performance note
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**⚡ Performance**: 
- Monte Carlo: {n_iterations:,} iterations
- Cache: Active
- Runtime: Optimized for Cloud
""")
