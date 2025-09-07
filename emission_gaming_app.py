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
**How companies can appear to meet Science-Based Targets through strategic emission factor selection**

This tool demonstrates how a Food & Beverage company can achieve apparent **4.2% annual emission reductions** 
(meeting SBTi requirements) through Scope 3 emission factor gaming alone—without any operational changes.
""")

# Key insight callout
st.info("**🚨 Key Finding**: Through factor selection, companies can appear SBTi-compliant for 13+ years while actually growing emissions")

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

# Main visualization section
st.subheader("🎮 Scope 3 Gaming Impact vs. SBTi Pathway")

# Calculate weighted factors for scenarios
scenarios = {
    "Conservative Selection": calculate_weighted_factor("Conservative"),
    "Moderate Selection": calculate_weighted_factor("Moderate"), 
    "Aggressive Selection": calculate_weighted_factor("Aggressive")
}

# Calculate gaming potential
total_gaming_potential = ((scenarios["Conservative Selection"] - scenarios["Aggressive Selection"]) / scenarios["Conservative Selection"]) * 100

# SBTi 4.2% annual reduction pathway
sbti_reduction_rate = 4.2  # Annual percentage reduction required

# Years for projection
years = list(range(2025, 2031))
base_year = 2025

# Calculate Scope 3 emissions trajectories with Monte Carlo for confidence intervals
@st.cache_data
def calculate_scope3_trajectories_with_ci(scenarios, production, growth, sbti_rate, n_iterations, uncertainty):
    np.random.seed(42)
    
    # Store all iterations for each scenario and year
    all_results = {scenario: {year: [] for year in years} for scenario in scenarios.keys()}
    
    for iteration in range(n_iterations):
        for scenario, base_factor in scenarios.items():
            for year in years:
                year_index = year - base_year
                
                # Production growth
                production_year = production * (1 + growth/100)**year_index
                
                # Add uncertainty to emission factor
                variation = np.random.uniform(-uncertainty/100, uncertainty/100)
                varied_factor = base_factor * (1 + variation)
                
                # Calculate Scope 3 emissions only
                scope_3_emissions = production_year * varied_factor
                
                all_results[scenario][year].append(scope_3_emissions)
    
    # Calculate statistics for each scenario and year
    trajectories_with_ci = {}
    for scenario in scenarios.keys():
        trajectories_with_ci[scenario] = {}
        for year in years:
            data = all_results[scenario][year]
            trajectories_with_ci[scenario][year] = {
                'mean': np.mean(data),
                'p2_5': np.percentile(data, 2.5),
                'p97_5': np.percentile(data, 97.5)
            }
    
    return trajectories_with_ci

# Calculate trajectories with confidence intervals
with st.spinner("Calculating confidence intervals..."):
    trajectories_ci = calculate_scope3_trajectories_with_ci(
        scenarios, annual_production, growth_rate, sbti_reduction_rate,
        n_iterations, uncertainty
    )

# Calculate SBTi pathway (based on conservative baseline)
sbti_data = []
base_emissions = trajectories_ci["Conservative Selection"][2025]['mean']
for year in years:
    year_index = year - base_year
    sbti_emissions = base_emissions * ((1 - sbti_reduction_rate/100)**year_index)
    sbti_data.append({'Year': year, 'SBTi_Emissions': sbti_emissions})

# Prepare data for Altair visualization with confidence intervals
chart_data_list = []

# Add confidence intervals and means for each scenario
scenario_colors = {
    'Conservative Selection': '#ff6b6b',
    'Moderate Selection': '#ffa500', 
    'Aggressive Selection': '#4ecdc4'
}

for year in years:
    for scenario in scenarios.keys():
        stats = trajectories_ci[scenario][year]
        
        # Add confidence interval data points
        chart_data_list.extend([
            {
                'Year': year,
                'Scenario': scenario,
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
        'Scenario': 'SBTi 4.2% Pathway',
        'Type': 'SBTi',
        'Emissions': sbti_value,
        'Lower_CI': sbti_value,
        'Upper_CI': sbti_value
    })

chart_df = pd.DataFrame(chart_data_list)

# Create Altair chart with confidence intervals
base_chart = alt.Chart(chart_df).add_selection(
    alt.selection_interval()
)

# Confidence intervals as bands
confidence_bands = base_chart.mark_area(
    opacity=0.2
).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Lower_CI:Q', title='Scope 3 Emissions (tCO₂e)', scale=alt.Scale(zero=False)),
    y2=alt.Y2('Upper_CI:Q'),
    color=alt.Color('Scenario:N', 
                   scale=alt.Scale(
                       domain=['Conservative Selection', 'Moderate Selection', 'Aggressive Selection', 'SBTi 4.2% Pathway'],
                       range=['#ff6b6b', '#ffa500', '#4ecdc4', '#45b7d1']
                   ),
                   legend=alt.Legend(title="95% Confidence Intervals"))
).transform_filter(
    alt.datum.Scenario != 'SBTi 4.2% Pathway'
)

# Mean lines
mean_lines = base_chart.mark_line(
    strokeWidth=3,
    point=alt.OverlayMarkDef(size=80)
).encode(
    x=alt.X('Year:O'),
    y=alt.Y('Emissions:Q'),
    color=alt.Color('Scenario:N',
                   scale=alt.Scale(
                       domain=['Conservative Selection', 'Moderate Selection', 'Aggressive Selection', 'SBTi 4.2% Pathway'],
                       range=['#ff6b6b', '#ffa500', '#4ecdc4', '#45b7d1']
                   ),
                   legend=alt.Legend(title="Mean Trajectories")),
    strokeDash=alt.condition(alt.datum.Scenario == 'SBTi 4.2% Pathway', alt.value([5, 5]), alt.value([0]))
)

# Combine charts
final_chart = (confidence_bands + mean_lines).resolve_scale(
    color='independent'
).properties(
    width=700,
    height=400,
    title=alt.TitleParams(
        text=['Scope 3 Gaming Impact vs. SBTi Pathway', 'with 95% Confidence Intervals'],
        fontSize=16,
        anchor='start'
    )
)

# Display the chart
st.altair_chart(final_chart, use_container_width=True)

# Store chart_data for metrics
chart_data = pd.DataFrame()
for year in years:
    row = {'Year': year}
    for scenario in scenarios.keys():
        row[scenario] = trajectories_ci[scenario][year]['mean']
    row['SBTi 4.2% Pathway'] = next(d for d in sbti_data if d['Year'] == year)['SBTi_Emissions']
    chart_data = pd.concat([chart_data, pd.DataFrame([row])], ignore_index=True)

chart_data = chart_data.set_index('Year')

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Gaming Potential", 
        f"{total_gaming_potential:.1f}%",
        help="Maximum apparent emission reduction through factor selection"
    )

with col2:
    # Calculate if aggressive gaming meets SBTi
    aggressive_2030 = chart_data.loc[2030, 'Aggressive Selection']
    conservative_2025 = chart_data.loc[2025, 'Conservative Selection']
    apparent_reduction = ((conservative_2025 - aggressive_2030) / conservative_2025) * 100
    
    st.metric(
        "Apparent Reduction (2030)",
        f"{apparent_reduction:.1f}%",
        help="Apparent emission reduction from 2025 baseline through gaming"
    )

with col3:
    # Years of SBTi compliance through gaming alone
    gaming_annual_rate = apparent_reduction / 5  # Over 5 years
    years_compliant = gaming_annual_rate / sbti_reduction_rate
    
    st.metric(
        "Gaming Compliance",
        f"{years_compliant:.1f} years",
        help="Years of apparent SBTi compliance through factor gaming alone"
    )

with col4:
    st.metric(
        "Portfolio Products",
        f"{len([p for p in product_mix.values() if p > 0])}",
        help="Number of products in portfolio mix"
    )

# Gaming mechanism explanation
st.subheader("🔍 How Scope 3 Gaming Works")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Scope 3 Gaming Process:**")
    st.markdown("""
    1. **Database Shopping**: Choose from multiple LCA databases
    2. **Strategic Selection**: Pick lowest available factors for purchased goods
    3. **Scope 3 Focus**: Target purchased goods & services emissions
    4. **Compound Effect**: Gaming scales with business growth
    5. **SBTi Compliance**: Appear to meet 4.2% annual reduction
    """)

with col_right:
    st.markdown("**Gaming vs. Reality:**")
    gaming_comparison = pd.DataFrame({
        'Metric': ['2030 Scope 3 Emissions (tCO₂e)', 'Reduction from 2025', 'SBTi Compliant?'],
        'Conservative Factors': [
            f"{chart_data.loc[2030, 'Conservative Selection']:,.0f}",
            f"{((chart_data.loc[2025, 'Conservative Selection'] - chart_data.loc[2030, 'Conservative Selection']) / chart_data.loc[2025, 'Conservative Selection'] * 100):.1f}%",
            "❌ No"
        ],
        'Aggressive Gaming': [
            f"{chart_data.loc[2030, 'Aggressive Selection']:,.0f}",
            f"{apparent_reduction:.1f}%",
            "✅ Yes"
        ]
    })
    st.dataframe(gaming_comparison, use_container_width=True)

# Monte Carlo Analysis
st.subheader("🎲 Monte Carlo Analysis")

@st.cache_data
def run_scope3_monte_carlo(scenarios, production, growth, n_iterations, uncertainty):
    np.random.seed(42)
    results = {scenario: [] for scenario in scenarios.keys()}
    
    for iteration in range(n_iterations):
        for scenario, base_factor in scenarios.items():
            # 2030 calculation with uncertainty
            production_2030 = production * (1 + growth/100)**5
            
            # Add uncertainty to emission factor
            variation = np.random.uniform(-uncertainty/100, uncertainty/100)
            varied_factor = base_factor * (1 + variation)
            
            # Calculate Scope 3 emissions only
            scope_3_emissions = production_2030 * varied_factor
            
            results[scenario].append(scope_3_emissions)
    
    return results

with st.spinner(f"Running {n_iterations:,} Monte Carlo iterations..."):
    mc_results = run_scope3_monte_carlo(
        scenarios, annual_production, growth_rate, n_iterations, uncertainty
    )

# Calculate statistics
mc_stats = {}
for scenario, data in mc_results.items():
    mc_stats[scenario] = {
        'mean': np.mean(data),
        'p5': np.percentile(data, 5),
        'p95': np.percentile(data, 95),
        'std': np.std(data)
    }

# Monte Carlo results
col_mc1, col_mc2 = st.columns(2)

with col_mc1:
    st.markdown("**2030 Scope 3 Emissions with Uncertainty (tCO₂e)**")
    mc_summary = []
    for scenario, stats in mc_stats.items():
        mc_summary.append({
            'Scenario': scenario,
            'Mean': f"{stats['mean']:,.0f}",
            '5th-95th Percentile': f"{stats['p5']:,.0f} - {stats['p95']:,.0f}"
        })
    
    df_mc = pd.DataFrame(mc_summary)
    st.dataframe(df_mc, use_container_width=True)

with col_mc2:
    st.markdown("**Statistical Gaming Analysis**")
    
    # Check confidence interval overlap
    conservative_range = [mc_stats["Conservative Selection"]["p5"], mc_stats["Conservative Selection"]["p95"]]
    aggressive_range = [mc_stats["Aggressive Selection"]["p5"], mc_stats["Aggressive Selection"]["p95"]]
    
    overlap = not (conservative_range[1] < aggressive_range[0] or aggressive_range[1] < conservative_range[0])
    
    st.write(f"**Gaming Effect Size**: {total_gaming_potential:.1f}%")
    st.write(f"**95% CI Overlap**: {'❌ No' if not overlap else '⚠️ Yes'}")
    st.write(f"**Statistical Significance**: {'✅ High' if not overlap else '⚠️ Moderate'}")

# Product-level breakdown
st.subheader("📊 Product-Level Gaming Breakdown")

# Create product gaming table
product_gaming_data = []
for product, factors in emission_factors.items():
    mix_pct = product_mix[product]
    contribution = (factors["Conservative"] * mix_pct/100) / sum(factors["Conservative"] * product_mix[p]/100 for p, factors in emission_factors.items()) * 100
    
    product_gaming_data.append({
        'Product': product,
        'Portfolio Mix (%)': f"{mix_pct:.1f}%",
        'Conservative Factor': factors["Conservative"],
        'Aggressive Factor': factors["Aggressive"],
        'Gaming Potential (%)': f"{factors['Gaming Potential']:.1f}%",
        'Contribution to Total': f"{contribution:.1f}%"
    })

df_products = pd.DataFrame(product_gaming_data)
st.dataframe(df_products, use_container_width=True)

# Product gaming visualization using Altair
st.markdown("**Gaming Potential by Product**")
gaming_data = pd.DataFrame({
    'Product': list(emission_factors.keys()),
    'Gaming Potential (%)': [factors['Gaming Potential'] for factors in emission_factors.values()]
})

gaming_chart = alt.Chart(gaming_data).mark_bar().encode(
    x=alt.X('Product:N', title='Product'),
    y=alt.Y('Gaming Potential (%):Q', title='Gaming Potential (%)'),
    color=alt.Color('Gaming Potential (%):Q', 
                   scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                   legend=None),
    tooltip=['Product:N', 'Gaming Potential (%):Q']
).properties(
    width=600,
    height=300
)

st.altair_chart(gaming_chart, use_container_width=True)

# Call to action and research context
st.markdown("---")
st.subheader("🔗 Research Context")

col_cta1, col_cta2 = st.columns(2)

with col_cta1:
    st.markdown("""
    **📄 Based on Published Research**
    
    *"Operationalizing corporate climate action through five research frontiers"*
    
    By Ramana Gudipudi et al., Institute for Sustainable Transition, European School of Management and Technology
    
    **Key Findings:**
    - Companies can reduce reported emissions by up to 6.7x through gaming
    - 85% of companies cite Scope 3 accounting as primary barrier
    - Current GHG Protocol enables strategic factor selection
    """)

with col_cta2:
    st.markdown("""
    **🎯 The Scope 3 Gaming Problem**
    
    **How it works:**
    - Purchased goods & services dominate F&B emissions
    - Multiple LCA databases offer different factors
    - Strategic selection can exceed SBTi requirements
    - No verification of factor choice rationale
    
    **Impact:**
    - Undermines SBTi credibility
    - Creates unfair competitive advantages
    - Enables large-scale greenwashing
    """)

# Technical details
st.markdown("---")
st.subheader("📈 Technical Details")

with st.expander("Methodology & Data Sources"):
    st.markdown("""
    **Emission Factors Sources:**
    - Milk: Poore & Nemecek (2018), Western Europe mixed systems, South Africa pasture-based
    - Cheese: Poore & Nemecek (2018), NRDC study (2024), Canadian dairy products (2013)  
    - Butter: Poore & Nemecek (2018), Journal of Dairy Science (2011), Canadian dairy products (2013)
    - Wheat: CarbonCloud Climate Hub (2023), India Punjab average, Finland/reduced tillage values
    - Rice: Carbon footprint of grain production in China (2017), India Punjab, Nature Reviews (2023)
    
    **Gaming Calculation:**
    - Portfolio weighted factors: Conservative (4.112 kg CO₂e/kg) vs Aggressive (1.597 kg CO₂e/kg)
    - Monte Carlo uncertainty: ±10% factor variation (1,000 iterations)
    - SBTi compliance: 4.2% annual absolute reduction requirement
    - Focus: Scope 3 purchased goods & services emissions only
    """)

# Footer
st.markdown("---")
st.markdown("""
**About this tool**: Demonstrates Scope 3 emission gaming using real data from peer-reviewed research. 
All emission factors and scenarios reflect actual variations found across major LCA databases.

*Research by Ramana Gudipudi, Luis Costa, Ponraj Arumugam, Matthew Agarwala, Jürgen P. Kropp, Felix Creutzig*
""")
