import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Emission Gaming Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("🎯 Emission Gaming Calculator")
st.markdown("""
**Discover how strategic emission factor selection can artificially manipulate corporate carbon footprints**

This interactive tool demonstrates the critical gap in current GHG Protocol guidance that enables 'emission gaming' - 
the strategic selection of emission factors that can artificially reduce reported emissions by up to 6.7 times.
""")

# Sidebar for controls
st.sidebar.header("Scenario Configuration")

# Sample emission factor data based on your research
emission_factors = {
    "Tomatoes": {
        "Agribalyse (France)": 0.83,
        "Ecoinvent (Global)": 1.42,
        "USDA LCA (USA)": 0.95,
        "UK Carbon Trust": 1.73,
        "Spanish LCA": 0.67
    },
    "Wheat": {
        "Agribalyse (France)": 0.60,
        "Ecoinvent (Global)": 0.82,
        "CarbonCloud (Nordic)": 0.35,
        "Indian Punjab": 1.20,
        "Canadian Average": 0.75
    },
    "Milk": {
        "Agribalyse (France)": 1.24,
        "Ecoinvent (Global)": 3.20,
        "South Africa Pasture": 0.60,
        "US Dairy Science": 2.80,
        "Nordic Average": 1.05
    },
    "Cheese": {
        "Agribalyse (France)": 9.78,
        "Ecoinvent (Global)": 13.50,
        "Canadian Dairy": 5.30,
        "UK Carbon Trust": 12.20,
        "NRDC Study": 8.90
    },
    "Rice": {
        "Agribalyse (France)": 1.20,
        "China Regional": 1.60,
        "Indian Punjab": 2.10,
        "IRRI Best Practice": 0.90,
        "Ecoinvent Global": 1.85
    }
}

# Product selection
selected_product = st.sidebar.selectbox(
    "Select Product",
    list(emission_factors.keys()),
    help="Choose a product to see emission factor variations across databases"
)

# Company settings
st.sidebar.subheader("Company Scenario")
company_size = st.sidebar.number_input(
    "Annual Production (tonnes)",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=1000
)

growth_rate = st.sidebar.slider(
    "Annual Growth Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=7.0,
    step=0.5
)

# Monte Carlo settings
st.sidebar.subheader("Simulation Settings")
n_iterations = st.sidebar.selectbox(
    "Monte Carlo Iterations",
    [100, 500, 1000],
    index=1
)

uncertainty_range = st.sidebar.slider(
    "Uncertainty Range (±%)",
    min_value=5.0,
    max_value=20.0,
    value=10.0,
    step=1.0
)

# Create two columns for the main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📊 Emission Factors for {selected_product}")
    st.markdown("*kg CO₂e per kg product*")
    
    # Create DataFrame for the selected product
    df_factors = pd.DataFrame(
        list(emission_factors[selected_product].items()),
        columns=['Database', 'Emission Factor']
    )
    df_factors = df_factors.sort_values('Emission Factor')
    
    # Display as bar chart using Streamlit
    st.bar_chart(df_factors.set_index('Database')['Emission Factor'])
    
    # Display the data table
    st.dataframe(df_factors, use_container_width=True)
    
    # Statistics
    factors = list(emission_factors[selected_product].values())
    min_factor = min(factors)
    max_factor = max(factors)
    variation = ((max_factor - min_factor) / min_factor) * 100
    
    st.metric("Variation Range", f"{variation:.1f}%")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Lowest Factor", f"{min_factor:.2f}")
    with col_b:
        st.metric("Highest Factor", f"{max_factor:.2f}")

with col2:
    st.subheader("🎮 Gaming Scenarios")
    
    # Gaming scenarios
    scenarios = {
        "Conservative Selection": max(factors),
        "Average Selection": np.mean(factors),
        "Aggressive Selection": min(factors)
    }
    
    # Calculate emissions for each scenario
    gaming_data = []
    for scenario, factor in scenarios.items():
        annual_emissions = company_size * factor
        reduction_vs_conservative = ((scenarios["Conservative Selection"] - factor) / scenarios["Conservative Selection"]) * 100
        gaming_data.append({
            'Scenario': scenario,
            'Emission Factor': f"{factor:.2f}",
            'Annual Emissions (tCO₂e)': f"{annual_emissions:,.0f}",
            'Apparent Reduction (%)': f"{max(0, reduction_vs_conservative):.1f}%"
        })
    
    df_gaming = pd.DataFrame(gaming_data)
    st.dataframe(df_gaming, use_container_width=True)
    
    # Simple bar chart for annual emissions
    emissions_chart_data = pd.DataFrame({
        'Scenario': [s['Scenario'] for s in gaming_data],
        'Emissions': [company_size * scenarios[s['Scenario']] for s in gaming_data]
    })
    st.bar_chart(emissions_chart_data.set_index('Scenario'))

# Multi-year projection
st.subheader("📈 5-Year Gaming Impact Simulation")

# Calculate projections
years = list(range(2025, 2031))
projection_data = []

for year in years:
    year_index = year - 2025
    production = company_size * (1 + growth_rate/100)**year_index
    
    row = {'Year': year}
    for scenario, factor in scenarios.items():
        emissions = production * factor
        row[scenario] = emissions
    projection_data.append(row)

df_projection = pd.DataFrame(projection_data)
df_projection_chart = df_projection.set_index('Year')

# Display line chart
st.line_chart(df_projection_chart)

# Display the data
st.dataframe(df_projection, use_container_width=True)

# Monte Carlo Simulation
st.subheader("🎲 Monte Carlo Analysis")

@st.cache_data
def run_simple_monte_carlo(product, company_size, growth_rate, n_iterations, uncertainty_range):
    """Simple Monte Carlo simulation"""
    
    factors = list(emission_factors[product].values())
    scenarios = {
        "Conservative": max(factors),
        "Average": np.mean(factors),
        "Aggressive": min(factors)
    }
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    results = {}
    
    for scenario, base_factor in scenarios.items():
        scenario_results = []
        
        for _ in range(n_iterations):
            # Calculate 5-year progression
            yearly_emissions = []
            for year_idx in range(6):  # 2025-2030
                volume = company_size * (1 + growth_rate/100)**year_idx
                # Add uncertainty
                variation = np.random.uniform(-uncertainty_range/100, uncertainty_range/100)
                varied_factor = base_factor * (1 + variation)
                emission = volume * varied_factor
                yearly_emissions.append(emission)
            
            scenario_results.append(yearly_emissions[-1])  # 2030 value
        
        results[scenario] = {
            'mean': np.mean(scenario_results),
            'std': np.std(scenario_results),
            'min': np.min(scenario_results),
            'max': np.max(scenario_results),
            'p5': np.percentile(scenario_results, 5),
            'p95': np.percentile(scenario_results, 95)
        }
    
    return results

with st.spinner(f"Running {n_iterations:,} Monte Carlo iterations..."):
    mc_results = run_simple_monte_carlo(
        selected_product, company_size, growth_rate, n_iterations, uncertainty_range
    )

# Display Monte Carlo results
st.subheader("📊 Monte Carlo Results (2030 Emissions)")

mc_summary = []
for scenario, stats in mc_results.items():
    mc_summary.append({
        'Scenario': scenario,
        'Mean (tCO₂e)': f"{stats['mean']:,.0f}",
        'Std Dev': f"{stats['std']:,.0f}",
        '5th Percentile': f"{stats['p5']:,.0f}",
        '95th Percentile': f"{stats['p95']:,.0f}"
    })

df_mc = pd.DataFrame(mc_summary)
st.dataframe(df_mc, use_container_width=True)

# Gaming impact analysis
st.subheader("💡 Key Gaming Insights")

col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)

# Calculate gaming potential
conservative_mean = mc_results["Conservative"]["mean"]
aggressive_mean = mc_results["Aggressive"]["mean"]
gaming_potential = ((conservative_mean - aggressive_mean) / conservative_mean) * 100

with col_insight1:
    st.metric(
        "Gaming Potential",
        f"{gaming_potential:.1f}%",
        help="Potential emission reduction through strategic factor selection"
    )

with col_insight2:
    absolute_impact = conservative_mean - aggressive_mean
    st.metric(
        "2030 Gaming Impact",
        f"{absolute_impact:,.0f} tCO₂e",
        help="Absolute difference in 2030 emissions"
    )

with col_insight3:
    # Check if confidence intervals overlap
    conservative_range = [mc_results["Conservative"]["p5"], mc_results["Conservative"]["p95"]]
    aggressive_range = [mc_results["Aggressive"]["p5"], mc_results["Aggressive"]["p95"]]
    
    overlap = not (conservative_range[1] < aggressive_range[0] or aggressive_range[1] < conservative_range[0])
    
    if not overlap:
        confidence = "High (No CI overlap)"
    else:
        confidence = "Moderate (CI overlap)"
    
    st.metric(
        "Statistical Confidence",
        confidence,
        help="Confidence in gaming effect significance"
    )

with col_insight4:
    st.metric(
        "Database Options",
        f"{len(emission_factors[selected_product])}",
        help="Number of different databases available"
    )

# Year-over-year gaming effect
st.subheader("📈 Gaming Effect Progression")

gaming_progression = []
for year in years:
    year_index = year - 2025
    production = company_size * (1 + growth_rate/100)**year_index
    
    conservative_emissions = production * scenarios["Conservative Selection"]
    aggressive_emissions = production * scenarios["Aggressive Selection"]
    
    gaming_effect = ((conservative_emissions - aggressive_emissions) / conservative_emissions) * 100
    absolute_difference = conservative_emissions - aggressive_emissions
    
    gaming_progression.append({
        'Year': year,
        'Gaming Effect (%)': gaming_effect,
        'Absolute Difference (tCO₂e)': absolute_difference
    })

df_gaming_progression = pd.DataFrame(gaming_progression)

# Show gaming effect over time
st.markdown("**Gaming Effect Over Time (%)**")
st.line_chart(df_gaming_progression.set_index('Year')['Gaming Effect (%)'])

# Show absolute difference
st.markdown("**Absolute Gaming Impact (tCO₂e)**")
st.line_chart(df_gaming_progression.set_index('Year')['Absolute Difference (tCO₂e)'])

# Display progression table
st.dataframe(df_gaming_progression, use_container_width=True)

# Call to action
st.markdown("---")
st.subheader("🔗 Learn More")

col_cta1, col_cta2 = st.columns(2)

with col_cta1:
    st.markdown("""
    **📄 Read the Research**
    
    *"Operationalizing corporate climate action through five research frontiers"*
    
    Discover how our proposed framework addresses emission gaming through:
    - AI-enhanced emission factor harmonization
    - Industry-specific materiality taxonomies  
    - Science-based verification protocols
    """)

with col_cta2:
    st.markdown("""
    **🎯 The Problem**
    
    Current GHG Protocol flexibility enables:
    - Strategic emission factor selection
    - Up to 6.7x variation in reported emissions
    - Undermined credibility in corporate climate action
    - Unfair competitive advantages through gaming
    """)

# Detailed explanation
st.markdown("---")
st.subheader("🔍 How Emission Gaming Works")

st.markdown("""
**The Gaming Process:**

1. **Multiple Databases Available**: Companies can choose from dozens of LCA databases (Agribalyse, Ecoinvent, USDA LCA, etc.)

2. **Factor Selection**: For any given product, emission factors can vary by 50-300% between databases

3. **Strategic Selection**: Companies can legally select factors that minimize their reported emissions

4. **Compound Effect**: Over time, with business growth, gaming effects compound significantly

5. **Credibility Crisis**: This creates unfair competitive advantages and undermines climate action credibility

**Example:** A food company producing 50,000 tonnes annually could report anywhere from {:.0f} to {:.0f} tCO₂e 
just by selecting different emission factors for {}—a {:.1f}% difference!
""".format(
    company_size * min(factors),
    company_size * max(factors),
    selected_product.lower(),
    ((max(factors) - min(factors)) / min(factors)) * 100
))

# Footer
st.markdown("---")
st.markdown("""
**About this tool:** This calculator demonstrates emission gaming using real emission factors from scientific literature. 
The scenarios reflect actual variations found across major LCA databases.

*Based on research by Ramana Gudipudi et al. - Institute for Sustainable Transition, European School of Management and Technology*

**Data sources:** Agribalyse, Ecoinvent, USDA LCA, various peer-reviewed studies
""")
