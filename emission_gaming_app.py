import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

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
    
    # Create bar chart
    fig_factors = px.bar(
        df_factors,
        x='Database',
        y='Emission Factor',
        title=f"Emission Factor Variations: {selected_product}",
        color='Emission Factor',
        color_continuous_scale='RdYlGn_r'
    )
    fig_factors.update_xaxis(tickangle=45)
    fig_factors.update_layout(height=400)
    st.plotly_chart(fig_factors, use_container_width=True)
    
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
    base_emissions = company_size * scenarios["Conservative Selection"]
    
    gaming_data = []
    for scenario, factor in scenarios.items():
        annual_emissions = company_size * factor
        reduction_vs_conservative = ((scenarios["Conservative Selection"] - factor) / scenarios["Conservative Selection"]) * 100
        gaming_data.append({
            'Scenario': scenario,
            'Emission Factor': factor,
            'Annual Emissions (tCO₂e)': annual_emissions,
            'Apparent Reduction (%)': max(0, reduction_vs_conservative)
        })
    
    df_gaming = pd.DataFrame(gaming_data)
    
    # Create emissions comparison chart
    fig_gaming = px.bar(
        df_gaming,
        x='Scenario',
        y='Annual Emissions (tCO₂e)',
        title="Gaming Impact on Annual Emissions",
        color='Apparent Reduction (%)',
        color_continuous_scale='RdYlGn',
        text='Annual Emissions (tCO₂e)'
    )
    fig_gaming.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_gaming.update_layout(height=400)
    st.plotly_chart(fig_gaming, use_container_width=True)

# Multi-year projection
st.subheader("📈 5-Year Gaming Impact Simulation")

years = list(range(2025, 2031))
production_volumes = [company_size * (1 + growth_rate/100)**i for i in range(6)]

# Create scenarios over time
fig_projection = go.Figure()

colors = ['red', 'orange', 'green']
for i, (scenario, factor) in enumerate(scenarios.items()):
    emissions = [vol * factor for vol in production_volumes]
    fig_projection.add_trace(go.Scatter(
        x=years,
        y=emissions,
        mode='lines+markers',
        name=scenario,
        line=dict(color=colors[i], width=3),
        marker=dict(size=8)
    ))

fig_projection.update_layout(
    title=f"Gaming Impact Over Time: {selected_product} ({growth_rate}% annual growth)",
    xaxis_title="Year",
    yaxis_title="Total Emissions (tCO₂e)",
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig_projection, use_container_width=True)

# Impact summary
st.subheader("💡 Key Insights")

col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)

with col_insight1:
    final_gaming_potential = gaming_progression[-1]['Gaming Effect (%)']
    st.metric(
        "Final Gaming Potential",
        f"{final_gaming_potential:.1f}%",
        help="Gaming potential in 2030 based on Monte Carlo simulation"
    )

with col_insight2:
    final_absolute_impact = gaming_progression[-1]['Absolute Difference (tCO₂e)']
    st.metric(
        "2030 Gaming Impact",
        f"{final_absolute_impact:,.0f} tCO₂e",
        help="Absolute emission difference in 2030 between gaming scenarios"
    )

with col_insight3:
    st.metric(
        "Statistical Confidence",
        f"{(1-p_value)*100:.1f}%",
        help="Confidence that gaming effects are statistically significant"
    )

with col_insight4:
    st.metric(
        "Database Variation",
        f"{len(emission_factors[selected_product])}x",
        help="Number of different databases providing factors for this product"
    )

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

# Footer
st.markdown("---")
st.markdown("""
**About this tool:** This calculator demonstrates emission gaming using real emission factors from scientific literature. 
The scenarios reflect actual variations found across major LCA databases (Agribalyse, Ecoinvent, USDA LCA, etc.).

*Based on research by Ramana Gudipudi et al. - Institute for Sustainable Transition, European School of Management and Technology*
""")

# Add some styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)