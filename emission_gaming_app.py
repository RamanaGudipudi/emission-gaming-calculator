import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Enable Altair to work with Streamlit
alt.data_transformers.enable('json')

# Page configuration
st.set_page_config(
    page_title="Scope 3 Emission Gaming Calculator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("🎯 Scope 3 Emission Gaming: The SBTi Compliance Loophole")
st.markdown("""
**How ONE company can appear to meet Science-Based Targets through strategic emission factor selection**

This tool demonstrates how a Food & Beverage company with a **shared 2025 baseline** can achieve apparent **4.2% annual emission reductions** 
(meeting SBTi requirements) through Scope 3 emission factor gaming alone—without any operational changes.
""")

# Key insight callout
st.info("**🚨 Key Finding**: Same company, same baseline → Strategic factor switching can show 13+ years of SBTi compliance while emissions actually grow")

# Sidebar configuration
st.sidebar.header("Company Configuration")

# Real emission factors from your research
emission_factors = {
    "Milk": {
        "Conservative": 3.20,
        "Moderate": 1.24, 
        "Aggressive": 0.60,
        "Gaming Potential": 81.3
    },
    "Cheese": {
        "Conservative": 13.50,
        "Moderate": 9.78,
        "Aggressive": 5.30,
        "Gaming Potential": 60.7
    },
    "Butter": {
        "Conservative": 12.00,
        "Moderate": 9.30,
        "Aggressive": 7.30,
        "Gaming Potential": 39.2
    },
    "Wheat": {
        "Conservative": 0.82,
        "Moderate": 0.60,
        "Aggressive": 0.35,
        "Gaming Potential": 57.3
    },
    "Rice": {
        "Conservative": 1.60,
        "Moderate": 1.20,
        "Aggressive": 0.90,
        "Gaming Potential": 43.8
    }
}

# Portfolio configuration
st.sidebar.subheader("Portfolio Mix (%)")
st.sidebar.markdown("*Adjust product composition*")

product_mix = {}
for product in emission_factors.keys():
    default_values = {"Milk": 30, "Cheese": 15, "Butter": 5, "Wheat": 35, "Rice": 15}
    product_mix[product] = st.sidebar.slider(
        f"{product}",
        min_value=0,
        max_value=50,
        value=default_values[product],
        step=1
    )

# Normalize to 100%
total_mix = sum(product_mix.values())
if total_mix > 0:
    product_mix = {k: (v/total_mix)*100 for k, v in product_mix.items()}

st.sidebar.markdown(f"**Total: {sum(product_mix.values()):.0f}%**")

# Company settings
st.sidebar.subheader("Company Scenario")
annual_production = st.sidebar.number_input(
    "Annual Production (tonnes)",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
    help="Total production volume across all products"
)

growth_rate = st.sidebar.slider(
    "Annual Growth Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=7.0,
    step=0.5,
    help="Business growth rate affecting production volumes"
)

# Gaming timeline configuration
st.sidebar.subheader("Gaming Timeline")
gaming_start_year = st.sidebar.selectbox(
    "Gaming Starts From",
    [2026, 2027, 2028],
    index=0,
    help="Year when strategic factor switching begins"
)

# Monte Carlo settings
st.sidebar.subheader("Simulation Settings")
n_iterations = st.sidebar.selectbox("Monte Carlo Iterations", [500, 1000, 2000], index=1)
uncertainty = st.sidebar.slider("Factor Uncertainty (±%)", 5.0, 15.0, 10.0, step=1.0)

# Calculate weighted emission factors
def calculate_weighted_factor(scenario):
    weighted_sum = 0
    for product, percentage in product_mix.items():
        factor = emission_factors[product][scenario]
        weighted_sum += factor * (percentage / 100)
    return weighted_sum

# Gaming strategy calculation
def get_emission_factor_for_year(year, strategy, baseline_factor):
    """
    Returns emission factor based on gaming strategy and timeline
    All strategies use conservative baseline in 2025
    """
    if year < gaming_start_year:
        # All use conservative factors for baseline convergence
        return baseline_factor
    else:
        # Apply gaming strategy from gaming_start_year onwards
        if strategy == "No Gaming (Honest)":
            return baseline_factor  # Stay with conservative
        elif strategy == "Moderate Gaming":
            return calculate_weighted_factor("Moderate")
        elif strategy == "Aggressive Gaming":
            return calculate_weighted_factor("Aggressive")
    return baseline_factor

# Main visualization section
st.subheader("🎮 Baseline Convergence → Strategic Factor Gaming")

# Calculate baseline factor (conservative for all)
baseline_factor = calculate_weighted_factor("Conservative")

# Gaming strategies
gaming_strategies = {
    "No Gaming (Honest)": "Conservative",
    "Moderate Gaming": "Moderate", 
    "Aggressive Gaming": "Aggressive"
}

# SBTi 4.2% annual reduction pathway
sbti_reduction_rate = 4.2  # Annual percentage reduction required

# Years for projection
years = list(range(2025, 2031))
base_year = 2025

# Calculate Scope 3 emissions trajectories with baseline convergence
@st.cache_data
def calculate_converged_trajectories_with_ci(gaming_strategies, baseline_factor, production, growth, sbti_rate, gaming_start, n_iterations, uncertainty):
    np.random.seed(42)
    
    # Store all iterations for each strategy and year
    all_results = {strategy: {year: [] for year in years} for strategy in gaming_strategies.keys()}
    
    for iteration in range(n_iterations):
        for strategy in gaming_strategies.keys():
            for year in years:
                year_index = year - base_year
                
                # Production growth
                production_year = production * (1 + growth/100)**year_index
                
                # Get emission factor based on gaming strategy and timeline
                emission_factor = get_emission_factor_for_year(year, strategy, baseline_factor)
                
                # Add uncertainty to emission factor
                variation = np.random.uniform(-uncertainty/100, uncertainty/100)
                varied_factor = emission_factor * (1 + variation)
                
                # Calculate Scope 3 emissions only
                scope_3_emissions = production_year * varied_factor
                
                all_results[strategy][year].append(scope_3_emissions)
    
    # Calculate statistics for each strategy and year
    trajectories_with_ci = {}
    for strategy in gaming_strategies.keys():
        trajectories_with_ci[strategy] = {}
        for year in years:
            data = all_results[strategy][year]
            trajectories_with_ci[strategy][year] = {
                'mean': np.mean(data),
                'p2_5': np.percentile(data, 2.5),
                'p97_5': np.percentile(data, 97.5)
            }
    
    return trajectories_with_ci

# Calculate trajectories with confidence intervals
with st.spinner("Calculating baseline convergence and gaming trajectories..."):
    trajectories_ci = calculate_converged_trajectories_with_ci(
        gaming_strategies, baseline_factor, annual_production, growth_rate, 
        sbti_reduction_rate, gaming_start_year, n_iterations, uncertainty
    )

# Calculate SBTi pathway (based on shared baseline)
sbti_data = []
base_emissions = trajectories_ci["No Gaming (Honest)"][2025]['mean']  # Shared baseline
for year in years:
    year_index = year - base_year
    sbti_emissions = base_emissions * ((1 - sbti_reduction_rate/100)**year_index)
    sbti_data.append({'Year': year, 'SBTi_Emissions': sbti_emissions})

# Prepare data for Altair visualization with confidence intervals
chart_data_list = []

# Gaming strategy colors
strategy_colors = {
    'No Gaming (Honest)': '#ff6b6b',      # Red - shows real emissions
    'Moderate Gaming': '#ffa500',          # Orange - moderate gaming
    'Aggressive Gaming': '#4ecdc4',        # Teal - maximum gaming
    'SBTi 4.2% Pathway': '#45b7d1'        # Blue - target pathway
}

for year in years:
    for strategy in gaming_strategies.keys():
        stats = trajectories_ci[strategy][year]
        
        # Add confidence interval data points
        chart_data_list.extend([
            {
                'Year': year,
                'Strategy': strategy,
                'Type': 'Mean',
                'Emissions': stats['mean'],
                'Lower_CI': stats['p2_5'],
                'Upper_CI': stats['p97_5']
            }
        ])
    
    # Add SBTi pathway
    sbti_value = next(d for d in sbti_data if d['Year'] == year)['SBTi_Emissions']
    chart_data_list.append({
        'Year': year,
        'Strategy': 'SBTi 4.2% Pathway',
        'Type': 'SBTi',
        'Emissions': sbti_value,
        'Lower_CI': sbti_value,
        'Upper_CI': sbti_value
    })

chart_df = pd.DataFrame(chart_data_list)

# Create Altair chart with baseline convergence visualization
base_chart = alt.Chart(chart_df).add_selection(
    alt.selection_interval()
)

# Confidence intervals as bands (excluding SBTi)
confidence_bands = base_chart.mark_area(
    opacity=0.2
).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Lower_CI:Q', title='Scope 3 Emissions (tCO₂e)', scale=alt.Scale(zero=False)),
    y2=alt.Y2('Upper_CI:Q'),
    color=alt.Color('Strategy:N', 
                   scale=alt.Scale(
                       domain=list(strategy_colors.keys()),
                       range=list(strategy_colors.values())
                   ),
                   legend=alt.Legend(title="95% Confidence Intervals"))
).transform_filter(
    alt.datum.Strategy != 'SBTi 4.2% Pathway'
)

# Mean lines with emphasis on gaming start
mean_lines = base_chart.mark_line(
    strokeWidth=3,
    point=alt.OverlayMarkDef(size=80)
).encode(
    x=alt.X('Year:O'),
    y=alt.Y('Emissions:Q'),
    color=alt.Color('Strategy:N',
                   scale=alt.Scale(
                       domain=list(strategy_colors.keys()),
                       range=list(strategy_colors.values())
                   ),
                   legend=alt.Legend(title="Gaming Strategies")),
    strokeDash=alt.condition(alt.datum.Strategy == 'SBTi 4.2% Pathway', alt.value([5, 5]), alt.value([0]))
)

# Add vertical line to show gaming start
gaming_start_line = alt.Chart(pd.DataFrame({'x': [gaming_start_year]})).mark_rule(
    color='red',
    strokeWidth=2,
    strokeDash=[3, 3]
).encode(
    x=alt.X('x:O'),
    tooltip=alt.value(f'Gaming Starts: {gaming_start_year}')
)

# Combine charts
final_chart = (confidence_bands + mean_lines + gaming_start_line).resolve_scale(
    color='independent'
).properties(
    width=700,
    height=400,
    title=alt.TitleParams(
        text=[f'Baseline Convergence (2025) → Gaming Starts ({gaming_start_year})', 
              'Same Company, Different Accounting Strategies'],
        fontSize=16,
        anchor='start'
    )
)

# Display the chart
st.altair_chart(final_chart, use_container_width=True)

# Store chart_data for metrics (using mean values)
chart_data = pd.DataFrame()
for year in years:
    row = {'Year': year}
    for strategy in gaming_strategies.keys():
        row[strategy] = trajectories_ci[strategy][year]['mean']
    row['SBTi 4.2% Pathway'] = next(d for d in sbti_data if d['Year'] == year)['SBTi_Emissions']
    chart_data = pd.concat([chart_data, pd.DataFrame([row])], ignore_index=True)

chart_data = chart_data.set_index('Year')

# Gaming effect analysis
st.subheader("📊 Gaming Impact Analysis")

# Calculate gaming metrics
baseline_emissions = chart_data.loc[2025, 'No Gaming (Honest)']  # Shared baseline
honest_2030 = chart_data.loc[2030, 'No Gaming (Honest)']
aggressive_2030 = chart_data.loc[2030, 'Aggressive Gaming']

# Real growth vs apparent reduction
real_growth = ((honest_2030 - baseline_emissions) / baseline_emissions) * 100
apparent_reduction = ((baseline_emissions - aggressive_2030) / baseline_emissions) * 100
gaming_effect = real_growth + apparent_reduction  # Total gaming impact

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Shared Baseline (2025)", 
        f"{baseline_emissions:,.0f} tCO₂e",
        help="All strategies start from the same baseline using conservative factors"
    )

with col2:
    st.metric(
        "Real Growth (2030)",
        f"+{real_growth:.1f}%",
        help="Actual emission increase from business growth (honest accounting)"
    )

with col3:
    st.metric(
        "Apparent Reduction",
        f"-{apparent_reduction:.1f}%",
        help="Reported emission reduction through aggressive factor gaming"
    )

with col4:
    # Years of SBTi compliance through gaming
    gaming_annual_rate = apparent_reduction / 5  # Over 5 years
    years_compliant = gaming_annual_rate / sbti_reduction_rate
    
    st.metric(
        "SBTi Compliance",
        f"{years_compliant:.1f} years",
        help="Years of apparent SBTi compliance through factor gaming alone"
    )

# Gaming comparison table
st.markdown("**Gaming vs. Reality Comparison (2030)**")
gaming_comparison = pd.DataFrame({
    'Accounting Strategy': ['Honest (Conservative)', 'Moderate Gaming', 'Aggressive Gaming', 'SBTi Target'],
    '2030 Emissions (tCO₂e)': [
        f"{chart_data.loc[2030, 'No Gaming (Honest)']:,.0f}",
        f"{chart_data.loc[2030, 'Moderate Gaming']:,.0f}",
        f"{chart_data.loc[2030, 'Aggressive Gaming']:,.0f}",
        f"{chart_data.loc[2030, 'SBTi 4.2% Pathway']:,.0f}"
    ],
    'Change from 2025': [
        f"+{real_growth:.1f}%",
        f"{((chart_data.loc[2030, 'Moderate Gaming'] - baseline_emissions) / baseline_emissions * 100):.1f}%",
        f"-{apparent_reduction:.1f}%",
        f"-{((baseline_emissions - chart_data.loc[2030, 'SBTi 4.2% Pathway']) / baseline_emissions * 100):.1f}%"
    ],
    'SBTi Compliant?': [
        "❌ No (Real emissions)",
        "⚠️ Partially",
        "✅ Yes (Through gaming)",
        "🎯 Target"
    ]
})

st.dataframe(gaming_comparison, use_container_width=True)

# Gaming mechanism explanation
st.subheader("🔍 The Baseline Convergence Gaming Mechanism")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Step-by-Step Gaming Process:**")
    st.markdown(f"""
    1. **Shared Baseline (2025)**: All use conservative factors → **{baseline_emissions:,.0f} tCO₂e**
    2. **Gaming Trigger ({gaming_start_year})**: Strategic factor switching begins
    3. **Database Shopping**: Choose most aggressive available factors  
    4. **Apparent Reduction**: Show -{apparent_reduction:.1f}% reduction by 2030
    5. **SBTi Compliance**: Meet 4.2% annual reduction target
    6. **Reality**: Actual emissions grew +{real_growth:.1f}% from business expansion
    """)

with col_right:
    st.markdown("**Why This Gaming Works:**")
    st.markdown("""
    ✅ **Same starting point** - baseline convergence is realistic
    
    ✅ **Plausible excuse** - "improved data quality" or "better methodology"
    
    ✅ **Regulatory approval** - SBTi accepts factor updates
    
    ✅ **No operational changes** required
    
    ✅ **Competitive advantage** over honest companies
    
    ❌ **Undermines climate credibility** and actual progress
    """)

# Timeline visualization
st.subheader("📅 Gaming Timeline")

timeline_data = []
for year in years:
    for strategy in gaming_strategies.keys():
        emissions = chart_data.loc[year, strategy]
        is_gaming_active = year >= gaming_start_year and strategy != "No Gaming (Honest)"
        
        timeline_data.append({
            'Year': year,
            'Strategy': strategy,
            'Emissions': emissions,
            'Gaming Active': is_gaming_active,
            'Phase': 'Baseline Period' if year < gaming_start_year else 'Gaming Period'
        })

timeline_df = pd.DataFrame(timeline_data)

# Create timeline chart
timeline_chart = alt.Chart(timeline_df).mark_circle(size=100).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Strategy:N', title='Accounting Strategy'),
    color=alt.Color('Gaming Active:N', 
                   scale=alt.Scale(domain=[True, False], range=['#ff4444', '#44ff44']),
                   legend=alt.Legend(title="Gaming Status")),
    size=alt.condition(alt.datum.Year == gaming_start_year, alt.value(200), alt.value(100)),
    tooltip=['Year:O', 'Strategy:N', 'Emissions:Q', 'Phase:N']
).properties(
    width=600,
    height=200,
    title=f'Gaming Timeline: Baseline Convergence → Strategic Switching ({gaming_start_year})'
)

st.altair_chart(timeline_chart, use_container_width=True)

# Monte Carlo Analysis with gaming focus
st.subheader("🎲 Statistical Gaming Analysis")

@st.cache_data
def run_gaming_monte_carlo(gaming_strategies, baseline_factor, production, growth, gaming_start, n_iterations, uncertainty):
    np.random.seed(42)
    results = {strategy: {'2025': [], '2030': []} for strategy in gaming_strategies.keys()}
    
    for iteration in range(n_iterations):
        for strategy in gaming_strategies.keys():
            # 2025 baseline (all same)
            baseline_production = production
            baseline_varied_factor = baseline_factor * (1 + np.random.uniform(-uncertainty/100, uncertainty/100))
            baseline_emissions = baseline_production * baseline_varied_factor
            results[strategy]['2025'].append(baseline_emissions)
            
            # 2030 with gaming
            production_2030 = production * (1 + growth/100)**5
            factor_2030 = get_emission_factor_for_year(2030, strategy, baseline_factor)
            varied_factor_2030 = factor_2030 * (1 + np.random.uniform(-uncertainty/100, uncertainty/100))
            emissions_2030 = production_2030 * varied_factor_2030
            results[strategy]['2030'].append(emissions_2030)
    
    return results

with st.spinner(f"Running {n_iterations:,} Monte Carlo iterations for gaming analysis..."):
    mc_gaming_results = run_gaming_monte_carlo(
        gaming_strategies, baseline_factor, annual_production, growth_rate, 
        gaming_start_year, n_iterations, uncertainty
    )

# Statistical significance of gaming effect
honest_2030_distribution = mc_gaming_results["No Gaming (Honest)"]['2030']
aggressive_2030_distribution = mc_gaming_results["Aggressive Gaming"]['2030']

# Calculate overlap and effect size
honest_mean = np.mean(honest_2030_distribution)
aggressive_mean = np.mean(aggressive_2030_distribution)
pooled_std = np.sqrt((np.var(honest_2030_distribution) + np.var(aggressive_2030_distribution)) / 2)
cohens_d = abs(honest_mean - aggressive_mean) / pooled_std

# Statistical significance test
from scipy import stats
t_stat, p_value = stats.ttest_ind(honest_2030_distribution, aggressive_2030_distribution)

col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.markdown("**Gaming Effect Statistics (2030)**")
    st.write(f"**Effect Size (Cohen's d)**: {cohens_d:.2f}")
    st.write(f"**Statistical Significance**: p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
    st.write(f"**Gaming Magnitude**: {((honest_mean - aggressive_mean) / honest_mean * 100):.1f}% difference")
    st.write(f"**Baseline Convergence**: ✅ Perfect (2025)")

with col_stat2:
    st.markdown("**Detectability Analysis**")
    overlap_threshold = 0.05  # 5% overlap threshold
    actual_overlap = len([x for x in aggressive_2030_distribution if min(honest_2030_distribution) <= x <= max(honest_2030_distribution)]) / len(aggressive_2030_distribution)
    
    st.write(f"**Distribution Overlap**: {actual_overlap*100:.1f}%")
    st.write(f"**Gaming Detectability**: {'🔍 Easily Detected' if actual_overlap < overlap_threshold else '⚠️ Hard to Detect'}")
    st.write(f"**Audit Red Flag**: {'🚨 High' if cohens_d > 1.0 else '⚠️ Medium' if cohens_d > 0.5 else '✅ Low'}")

# Call to action and research context
st.markdown("---")
st.subheader("🔗 Research Context & Implications")

col_cta1, col_cta2 = st.columns(2)

with col_cta1:
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

with col_cta2:
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

# Technical details
st.markdown("---")
st.subheader("📈 Technical Methodology")

with st.expander("Gaming Timeline & Statistical Analysis"):
    st.markdown(f"""
    **Baseline Convergence Approach:**
    - **2025 Baseline**: All strategies use conservative factors → {baseline_factor:.3f} kg CO₂e/kg weighted average
    - **Gaming Start**: {gaming_start_year} → Strategic factor switching begins
    - **Shared Reality**: Same business growth rate ({growth_rate}%) across all strategies
    - **Gaming Effect**: Apparent reductions while actual emissions grow
    
    **Statistical Validation:**
    - **Monte Carlo Iterations**: {n_iterations:,} simulations with ±{uncertainty}% factor uncertainty
    - **Effect Size**: Cohen's d = {cohens_d:.2f} ({"Large" if cohens_d > 0.8 else "Medium" if cohens_d > 0.5 else "Small"} effect)
    - **Significance**: p < 0.001 (highly significant difference between honest vs gaming)
    - **Gaming Magnitude**: {((honest_mean - aggressive_mean) / honest_mean * 100):.1f}% emissions difference by 2030
    
    **Real-World Context:**
    - Gaming timeline mirrors actual corporate reporting cycles
    - Factor databases span {min([min(factors.values()) for factors in emission_factors.values()]):.2f} - {max([max(factors.values()) for factors in emission_factors.values()]):.2f} kg CO₂e/kg
    - SBTi compliance threshold: 4.2% annual absolute reduction
    """)

# Footer
st.markdown("---")
st.markdown(f"""
**About this tool**: Demonstrates baseline convergence gaming using real emission factor variations from peer-reviewed research. 
Shows how ONE company can strategically switch factors starting from {gaming_start_year} to appear SBTi-compliant while emissions actually grow.

*Research by Ramana Gudipudi, Luis Costa, Ponraj Arumugam, Matthew Agarwala, Jürgen P. Kropp, Felix Creutzig*
""")
