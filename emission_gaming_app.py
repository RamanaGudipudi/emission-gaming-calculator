import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# Create Plotly visualization with confidence intervals
fig = go.Figure()

# Color scheme
colors = {
    'Conservative Selection': '#ff6b6b',  # Red
    'Moderate Selection': '#ffa500',      # Orange  
    'Aggressive Selection': '#4ecdc4',    # Teal
    'SBTi 4.2% Pathway': '#45b7d1'       # Blue
}

# Add confidence intervals as filled areas
for scenario in ['Conservative Selection', 'Moderate Selection', 'Aggressive Selection']:
    upper_values = [trajectories_ci[scenario][year]['p97_5'] for year in years]
    lower_values = [trajectories_ci[scenario][year]['p2_5'] for year in years]
    
    # Add upper bound (invisible line)
    fig.add_trace(go.Scatter(
        x=years,
        y=upper_values,
        mode='lines',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add lower bound and fill to upper
    fig.add_trace(go.Scatter(
        x=years,
        y=lower_values,
        mode='lines',
        fill='tonexty',
        fillcolor=colors[scenario].replace('#', 'rgba(') + ', 0.2)'.replace(')', ',0.2)'),
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{scenario} 95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))

# Add mean lines
for scenario in scenarios.keys():
    mean_values = [trajectories_ci[scenario][year]['mean'] for year in years]
    
    fig.add_trace(go.Scatter(
        x=years,
        y=mean_values,
        mode='lines+markers',
        line=dict(color=colors[scenario], width=3),
        marker=dict(size=8),
        name=f'{scenario} (Mean)',
        hovertemplate=f'{scenario}<br>Year: %{{x}}<br>Emissions: %{{y:,.0f}} tCO₂e<extra></extra>'
    ))

# Add SBTi pathway
sbti_values = [d['SBTi_Emissions'] for d in sbti_data]
fig.add_trace(go.Scatter(
    x=years,
    y=sbti_values,
    mode='lines+markers',
    line=dict(color=colors['SBTi 4.2% Pathway'], width=3, dash='dash'),
    marker=dict(size=8, symbol='square'),
    name='SBTi 4.2% Pathway',
    hovertemplate='SBTi Pathway<br>Year: %{x}<br>Emissions: %{y:,.0f} tCO₂e<extra></extra>'
))

# Update layout
fig.update_layout(
    title={
        'text': 'Scope 3 Gaming Impact vs. SBTi Pathway<br><sub>with 95% Confidence Intervals</sub>',
        'x': 0.5,
        'font': {'size': 16}
    },
    xaxis_title='Year',
    yaxis_title='Scope 3 Emissions (tCO₂e)',
    height=600,
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.01
    )
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Create chart_data for metrics
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

# Product gaming visualization
st.markdown("**Gaming Potential by Product**")
gaming_chart_data = pd.DataFrame({
    'Product': list(emission_factors.keys()),
    'Gaming Potential (%)': [factors['Gaming Potential'] for factors in emission_factors.values()]
})
gaming_chart_data = gaming_chart_data.set_index('Product')
st.bar_chart(gaming_chart_data)

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
