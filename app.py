"""
================================================================================
KAIROS THERAPEUTICS - THERAPEUTIC TARGET EXPLORER
================================================================================
Investor Demo | ML Prototype V0.6

Single-file Streamlit application for interactive exploration of therapeutic
targets identified by the Kairos AI-guided discovery platform.

Run: streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Optional imports with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
APP_VERSION = "v0.6-public-demo"
APP_TITLE = "Kairos Therapeutics | Target Explorer"

# Color scheme
COLORS = {
    'tier1': '#2ECC71',      # Green
    'tier2': '#F39C12',      # Orange
    'tier3': '#95A5A6',      # Gray
    'primary': '#1E3A5F',    # Dark blue
    'secondary': '#5A6C7D',  # Slate
    'up': '#E74C3C',         # Red (disease elevated)
    'down': '#3498DB',       # Blue (disease decreased)
}

# =============================================================================
# PAGE CONFIG (must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    /* Metric improvements */
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #1E3A5F;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #F0F2F6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Pipeline step cards */
    .pipeline-step {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        height: 140px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """
    Load prioritized targets from CSV with graceful fallback.
    
    Search order:
    1. data/processed/prioritized_targets.csv (full data)
    2. data/processed/prioritized_targets_streamlit.csv (simplified)
    3. Synthetic demo data (fallback)
    """
    
    # Possible file locations (relative to app.py)
    search_paths = [
        Path("data/processed/prioritized_targets.csv"),
        Path("data/processed/prioritized_targets_streamlit.csv"),
        Path("./prioritized_targets.csv"),
    ]
    
    for path in search_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                # Validate minimum required columns
                required = ['gene', 'priority_score', 'priority_tier', 'direction']
                if all(col in df.columns for col in required):
                    return df, str(path), False
            except Exception as e:
                continue
    
    # Fallback to synthetic demo data
    return create_synthetic_data(), "Synthetic Demo Data", True

def create_synthetic_data():
    """Generate synthetic demo data for public demonstrations."""
    np.random.seed(42)
    
    # Anonymous target names
    n_targets = 19
    genes = [f"TARGET_{chr(65+i)}" for i in range(n_targets)]
    
    # Realistic score distribution
    scores = np.linspace(95, 42, n_targets) + np.random.normal(0, 3, n_targets)
    scores = np.clip(scores, 40, 100).round(1)
    scores = sorted(scores, reverse=True)
    
    # Assign tiers based on scores
    tiers = []
    for s in scores:
        if s >= 70:
            tiers.append('Tier 1: Lead')
        elif s >= 50:
            tiers.append('Tier 2: Development')
        else:
            tiers.append('Tier 3: Research')
    
    # Generate other fields
    directions = np.random.choice(['Up in Disease', 'Down in Disease'], n_targets, p=[0.75, 0.25])
    log2fc = np.where(directions == 'Up in Disease',
                      np.random.uniform(1.0, 3.5, n_targets),
                      np.random.uniform(-3.0, -1.0, n_targets)).round(2)
    
    druggability = np.random.choice(['High', 'Medium', 'Low'], n_targets, p=[0.25, 0.45, 0.30])
    
    hallmarks = np.random.choice([
        'ECM & Tissue Integrity',
        'Angiogenesis / Trophic', 
        'Inflammation / Immune',
        'Senescence / Cell Cycle',
        'Other / Mixed'
    ], n_targets, p=[0.35, 0.25, 0.15, 0.10, 0.15])
    
    return pd.DataFrame({
        'rank': range(1, n_targets + 1),
        'gene': genes,
        'priority_score': scores,
        'priority_tier': tiers,
        'direction': directions,
        'log2FC': log2fc,
        'druggability': druggability,
        'aging_hallmark': hallmarks,
        'msc_strategy': ['MSC-delivered intervention'] * n_targets,
        'risk_flag': np.random.choice(['Low', 'Evaluate carefully', 'Context-dependent'], 
                                       n_targets, p=[0.75, 0.20, 0.05])
    })

# Load data
df, data_source, is_synthetic = load_data()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    # Logo placeholder
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1E3A5F; margin: 0;">🧬</h1>
        <h2 style="color: #1E3A5F; margin: 0; font-size: 1.3rem;">Kairos Therapeutics</h2>
        <p style="color: #5A6C7D; margin: 0; font-size: 0.9rem;">Target Explorer</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data source indicator
    if is_synthetic:
        st.warning("⚠️ **Demo Mode**\nUsing synthetic data")
    else:
        st.success(f"✅ **Live Data**\n{data_source}")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filters")
    
    # Tier filter
    tier_options = ['All Tiers'] + sorted(df['priority_tier'].unique().tolist())
    selected_tier = st.selectbox("Priority Tier", tier_options, key="tier_filter")
    
    # Direction filter
    direction_options = ['All Directions'] + sorted(df['direction'].unique().tolist())
    selected_direction = st.selectbox("Direction", direction_options, key="direction_filter")
    
    # Druggability filter (if column exists)
    if 'druggability' in df.columns:
        drug_options = ['All Druggability'] + sorted(df['druggability'].unique().tolist())
        selected_drug = st.selectbox("Druggability", drug_options, key="drug_filter")
    else:
        selected_drug = 'All Druggability'
    
    # Hallmark filter (if column exists)
    if 'aging_hallmark' in df.columns:
        hallmark_options = ['All Hallmarks'] + sorted(df['aging_hallmark'].unique().tolist())
        selected_hallmark = st.selectbox("Aging Hallmark", hallmark_options, key="hallmark_filter")
    else:
        selected_hallmark = 'All Hallmarks'
    
    st.markdown("---")
    
    # Info section
    st.markdown("### ℹ️ About")
    st.markdown(f"""
    **Pipeline:** ML Prototype V0  
    **Version:** `{APP_VERSION}`  
    **Dataset:** GSE114007  
    **Focus:** OA × Aging Biology
    """)
    
    st.markdown("---")
    st.markdown(f"<p style='text-align:center; color:#95A5A6; font-size:0.8rem;'>{APP_VERSION}</p>", 
                unsafe_allow_html=True)

# =============================================================================
# APPLY FILTERS
# =============================================================================
filtered_df = df.copy()

if selected_tier != 'All Tiers':
    filtered_df = filtered_df[filtered_df['priority_tier'] == selected_tier]
if selected_direction != 'All Directions':
    filtered_df = filtered_df[filtered_df['direction'] == selected_direction]
if selected_drug != 'All Druggability' and 'druggability' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['druggability'] == selected_drug]
if selected_hallmark != 'All Hallmarks' and 'aging_hallmark' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['aging_hallmark'] == selected_hallmark]

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<p class="main-header">🧬 Therapeutic Target Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Guided Discovery Platform | Kairos Therapeutics</p>', unsafe_allow_html=True)

# METHODOLOGY DEMO badge (only in synthetic mode)
if is_synthetic:
    st.markdown("""
    <div style="margin: 0.5rem 0 1rem 0;">
        <span style="background-color: #E74C3C; color: white; padding: 0.3rem 1rem; 
                     border-radius: 20px; font-size: 0.85rem; font-weight: 600; 
                     letter-spacing: 0.5px;">
            METHODOLOGY DEMO
        </span>
        <span style="color: #7F8C8D; font-size: 0.85rem; margin-left: 0.8rem;">
            Synthetic data for public demonstration
        </span>
    </div>
    """, unsafe_allow_html=True)

# Key Metrics Row
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    tier1_count = len(df[df['priority_tier'] == 'Tier 1: Lead'])
    st.metric("🥇 Tier 1 Leads", tier1_count)

with col2:
    st.metric("🎯 Total Targets", len(df))

with col3:
    if 'druggability' in df.columns:
        high_drug = len(df[df['druggability'] == 'High'])
        st.metric("💊 High Druggability", high_drug)
    else:
        st.metric("💊 High Druggability", "N/A")

with col4:
    inhibition = len(df[df['direction'].str.contains('Up', na=False)])
    st.metric("🔻 Inhibition", inhibition)

with col5:
    restoration = len(df[df['direction'].str.contains('Down', na=False)])
    st.metric("🔺 Restoration", restoration)

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Target Table", 
    "📊 Visualizations", 
    "🔬 Gene Deep Dive", 
    "📈 Pipeline Overview"
])

# -----------------------------------------------------------------------------
# TAB 1: TARGET TABLE
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### Prioritized Therapeutic Targets")
    st.markdown(f"Showing **{len(filtered_df)}** of {len(df)} targets")
    
    # Select display columns
    display_cols = ['rank', 'gene', 'priority_score', 'priority_tier', 'direction', 'log2FC']
    optional_cols = ['druggability', 'aging_hallmark', 'risk_flag', 'msc_strategy']
    for col in optional_cols:
        if col in filtered_df.columns:
            display_cols.append(col)
    
    # Display dataframe
    display_df = filtered_df[[c for c in display_cols if c in filtered_df.columns]].copy()
    
    # Format numeric columns
    if 'priority_score' in display_df.columns:
        display_df['priority_score'] = display_df['priority_score'].round(1)
    if 'log2FC' in display_df.columns:
        display_df['log2FC'] = display_df['log2FC'].round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        column_config={
            'rank': st.column_config.NumberColumn('Rank', format='%d'),
            'gene': st.column_config.TextColumn('Gene', width='medium'),
            'priority_score': st.column_config.ProgressColumn(
                'Priority Score',
                min_value=0,
                max_value=100,
                format='%.1f'
            ),
            'priority_tier': st.column_config.TextColumn('Tier', width='medium'),
            'direction': st.column_config.TextColumn('Direction', width='medium'),
            'log2FC': st.column_config.NumberColumn('log2FC', format='%.2f'),
            'druggability': st.column_config.TextColumn('Druggability', width='small'),
            'aging_hallmark': st.column_config.TextColumn('Hallmark', width='medium'),
        }
    )
    
    # Download button
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"kairos_targets_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 2: VISUALIZATIONS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Target Prioritization Visualizations")
    
    if not PLOTLY_AVAILABLE:
        st.error("⚠️ Plotly not installed. Run: `pip install plotly`")
    else:
        viz_row1_col1, viz_row1_col2 = st.columns(2)
        
        with viz_row1_col1:
            # Priority Score Bar Chart
            chart_df = filtered_df.head(15).copy()
            fig_bar = px.bar(
                chart_df,
                x='priority_score',
                y='gene',
                orientation='h',
                color='priority_tier',
                color_discrete_map={
                    'Tier 1: Lead': COLORS['tier1'],
                    'Tier 2: Development': COLORS['tier2'],
                    'Tier 3: Research': COLORS['tier3']
                },
                title='<b>Target Priority Ranking</b>',
                labels={'priority_score': 'Priority Score', 'gene': 'Target'}
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=True,
                legend_title_text='Tier',
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_bar.add_vline(x=70, line_dash="dash", line_color=COLORS['tier1'], 
                             annotation_text="Tier 1", annotation_position="top right")
            fig_bar.add_vline(x=50, line_dash="dash", line_color=COLORS['tier2'],
                             annotation_text="Tier 2", annotation_position="top right")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_row1_col2:
            # Tier Distribution
            tier_counts = df['priority_tier'].value_counts()
            fig_pie = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                color=tier_counts.index,
                color_discrete_map={
                    'Tier 1: Lead': COLORS['tier1'],
                    'Tier 2: Development': COLORS['tier2'],
                    'Tier 3: Research': COLORS['tier3']
                },
                title='<b>Target Distribution by Tier</b>',
                hole=0.4
            )
            fig_pie.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        viz_row2_col1, viz_row2_col2 = st.columns(2)
        
        with viz_row2_col1:
            # Scatter: Priority vs Effect Size
            scatter_df = filtered_df.copy()
            scatter_df['abs_log2FC'] = scatter_df['log2FC'].abs()
            
            fig_scatter = px.scatter(
                scatter_df,
                x='abs_log2FC',
                y='priority_score',
                color='priority_tier',
                size='priority_score',
                hover_name='gene',
                hover_data=['direction', 'log2FC'],
                color_discrete_map={
                    'Tier 1: Lead': COLORS['tier1'],
                    'Tier 2: Development': COLORS['tier2'],
                    'Tier 3: Research': COLORS['tier3']
                },
                title='<b>Priority Score vs Effect Size</b>',
                labels={'abs_log2FC': '|log2FC| (Effect Size)', 'priority_score': 'Priority Score'}
            )
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            fig_scatter.add_hline(y=70, line_dash="dash", line_color=COLORS['tier1'], opacity=0.5)
            fig_scatter.add_hline(y=50, line_dash="dash", line_color=COLORS['tier2'], opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with viz_row2_col2:
            # Hallmark distribution
            if 'aging_hallmark' in df.columns:
                hallmark_counts = df['aging_hallmark'].value_counts()
                fig_hallmark = px.bar(
                    x=hallmark_counts.values,
                    y=hallmark_counts.index,
                    orientation='h',
                    title='<b>Targets by Aging Hallmark</b>',
                    labels={'x': 'Number of Targets', 'y': 'Hallmark'},
                    color=hallmark_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_hallmark.update_layout(
                    height=400, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_hallmark, use_container_width=True)
            else:
                st.info("Hallmark data not available")

# -----------------------------------------------------------------------------
# TAB 3: GENE DEEP DIVE
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔬 Gene Deep Dive")
    st.markdown("Select a target to view its complete therapeutic profile.")
    
    # Gene selector
    gene_list = df.sort_values('rank')['gene'].tolist()
    selected_gene = st.selectbox(
        "Select Target:",
        options=gene_list,
        format_func=lambda x: f"{x} (Rank #{df[df['gene']==x]['rank'].values[0]})"
    )
    
    # Get gene data
    gene_data = df[df['gene'] == selected_gene].iloc[0]
    
    st.markdown("---")
    
    # Gene header
    tier_color = COLORS['tier1'] if 'Tier 1' in gene_data['priority_tier'] else \
                 COLORS['tier2'] if 'Tier 2' in gene_data['priority_tier'] else COLORS['tier3']
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <h1 style="margin: 0; color: {COLORS['primary']};">{selected_gene}</h1>
        <span style="background-color: {tier_color}; color: white; padding: 0.3rem 1rem; 
                     border-radius: 20px; font-weight: 600;">
            {gene_data['priority_tier']}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Three-column layout
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    
    with prof_col1:
        st.markdown("#### 📊 Priority Metrics")
        st.metric("Priority Score", f"{gene_data['priority_score']:.1f} / 100")
        st.metric("Overall Rank", f"#{int(gene_data['rank'])} of {len(df)}")
        
        if 'risk_flag' in gene_data.index:
            risk = gene_data['risk_flag']
            if risk == 'Low':
                st.success(f"**Risk:** {risk} ✓")
            elif risk == 'Evaluate carefully':
                st.warning(f"**Risk:** {risk}")
            else:
                st.error(f"**Risk:** {risk}")
    
    with prof_col2:
        st.markdown("#### 🧬 Expression Profile")
        
        log2fc = gene_data['log2FC']
        direction = gene_data['direction']
        
        st.metric("log2 Fold Change", f"{log2fc:.2f}")
        
        if 'Up' in direction:
            st.error(f"**{direction}**")
            st.markdown("➡️ **Strategy:** Inhibition target")
        else:
            st.info(f"**{direction}**")
            st.markdown("➡️ **Strategy:** Restoration target")
        
        if 'aging_hallmark' in gene_data.index:
            st.markdown(f"**Hallmark:** {gene_data['aging_hallmark']}")
    
    with prof_col3:
        st.markdown("#### 💊 Therapeutic Profile")
        
        if 'druggability' in gene_data.index:
            drug = gene_data['druggability']
            if drug == 'High':
                st.success(f"**Druggability:** {drug}")
            elif drug == 'Medium':
                st.warning(f"**Druggability:** {drug}")
            else:
                st.info(f"**Druggability:** {drug}")
        
        if 'msc_strategy' in gene_data.index and pd.notna(gene_data['msc_strategy']):
            st.markdown("**MSC Delivery Strategy:**")
            st.markdown(f"_{gene_data['msc_strategy']}_")
        
        if 'existing_drugs' in gene_data.index and pd.notna(gene_data['existing_drugs']):
            st.markdown(f"**Existing Drugs:** {gene_data['existing_drugs']}")
    
    # Therapeutic rationale
    st.markdown("---")
    st.markdown("#### 🎯 Therapeutic Rationale")
    
    if 'Up' in gene_data['direction']:
        rationale = f"""
        **{selected_gene}** is **elevated in disease** (log2FC = {gene_data['log2FC']:.2f}), 
        indicating it contributes to pathological processes.
        
        **Intervention Strategy:** Inhibition / Blocking
        - MSC-delivered inhibitor or antagonist
        - Neutralizing antibody approach
        - Small molecule intervention (if druggable)
        
        **Expected Outcome:** Reducing {selected_gene} activity may restore healthy tissue 
        homeostasis by blocking disease-associated signaling cascades.
        """
    else:
        rationale = f"""
        **{selected_gene}** is **decreased in disease** (log2FC = {gene_data['log2FC']:.2f}), 
        indicating loss of a protective factor.
        
        **Intervention Strategy:** Restoration / Supplementation
        - MSC-delivered recombinant factor
        - MSC overexpression engineering
        - Gene therapy approach
        
        **Expected Outcome:** Restoring {selected_gene} levels may recover protective 
        signaling lost during disease progression.
        """
    st.markdown(rationale)

# -----------------------------------------------------------------------------
# TAB 4: PIPELINE OVERVIEW
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📈 Kairos ML Pipeline Overview")
    
    # Investor value proposition box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%); 
                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
        <h4 style="margin: 0 0 0.8rem 0; font-size: 1.1rem;">💡 Investment Thesis</h4>
        <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">
            <strong>Kairos Therapeutics</strong> is building an AI-guided platform to discover 
            therapeutic targets at the intersection of <strong>aging biology</strong> and 
            <strong>disease pathology</strong>. We treat aging as the root cause — diseases 
            are manifestations.
        </p>
        <p style="margin: 0.8rem 0 0 0; font-size: 0.95rem; line-height: 1.6;">
            <strong>Why Osteoarthritis?</strong> OA is an ideal disease wedge: aging-driven, 
            high unmet need (500M+ patients), and treatable via MSC-delivered interventions 
            — our core therapeutic modality.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    The platform identifies therapeutic targets that can be delivered via engineered 
    mesenchymal stem cells (MSCs), creating a systematic approach to age-related diseases.
    """)
    
    st.markdown("---")
    st.markdown("### 🔄 Discovery Pipeline")
    
    # Pipeline steps
    pipe_col1, pipe_col2, pipe_col3, pipe_col4 = st.columns(4)
    
    with pipe_col1:
        st.markdown(f"""
        <div class="pipeline-step" style="background: linear-gradient(135deg, #3498DB, #2980B9);">
            <h4 style="margin-top:0;">📥 Data Ingestion</h4>
            <p style="font-size: 0.85rem; margin: 0;">
                GEO datasets<br>
                Quality control<br>
                Normalization
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Notebooks 01-02")
    
    with pipe_col2:
        st.markdown(f"""
        <div class="pipeline-step" style="background: linear-gradient(135deg, #9B59B6, #8E44AD);">
            <h4 style="margin-top:0;">🔬 DE Analysis</h4>
            <p style="font-size: 0.85rem; margin: 0;">
                Differential expression<br>
                Pathway enrichment<br>
                Hallmark mapping
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Notebooks 03-04")
    
    with pipe_col3:
        st.markdown(f"""
        <div class="pipeline-step" style="background: linear-gradient(135deg, #E74C3C, #C0392B);">
            <h4 style="margin-top:0;">🎯 Intersection</h4>
            <p style="font-size: 0.85rem; margin: 0;">
                Disease × Aging<br>
                Secretome filter<br>
                Risk annotation
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Notebook 05")
    
    with pipe_col4:
        st.markdown(f"""
        <div class="pipeline-step" style="background: linear-gradient(135deg, #2ECC71, #27AE60);">
            <h4 style="margin-top:0;">📊 Prioritization</h4>
            <p style="font-size: 0.85rem; margin: 0;">
                Multi-factor scoring<br>
                Druggability<br>
                Tier assignment
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Notebook 06")
    
    st.markdown("---")
    
    # Key findings
    st.markdown("### 📋 Key Findings Summary")
    
    findings_col1, findings_col2 = st.columns(2)
    
    with findings_col1:
        st.markdown("""
        **Disease Biology (OA):**
        - Significant ECM degradation (MMP elevation)
        - Loss of protective angiogenic factors (VEGF↓)
        - Chronic inflammatory signaling
        - Fibrotic matrix remodeling
        
        **Aging Intersection:**
        - 34 genes at disease-aging overlap
        - 19 MSC-deliverable candidates identified
        - 10 Tier 1 lead targets prioritized
        """)
    
    with findings_col2:
        st.markdown("""
        **Therapeutic Strategies:**
        
        | Strategy | Count | Approach |
        |----------|-------|----------|
        | Inhibition | 15 | Block disease-elevated factors |
        | Restoration | 4 | Replace protective factors |
        
        **Top Target Classes:**
        - Matrix metalloproteinases (MMPs)
        - Growth factor pathways (VEGF, IGF)
        - ECM remodeling proteins
        """)
    
    st.markdown("---")
    
    # Resources
    st.markdown("### 🔗 Resources")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.markdown("""
        **📂 Repository**  
        [GitHub: kairostx/Kairos_Therapeutics](https://github.com/kairostx/Kairos_Therapeutics)
        """)
    
    with res_col2:
        st.markdown("""
        **📊 Dataset**  
        [GEO: GSE114007](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114007)
        """)
    
    with res_col3:
        st.markdown(f"""
        **🏷️ Version**  
        `{APP_VERSION}`
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem 0; color: #95A5A6;">
    <p style="margin: 0; font-size: 1rem;">
        🧬 <strong style="color: #1E3A5F;">Kairos Therapeutics</strong>
    </p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem;">
        {APP_VERSION}
    </p>
</div>
""", unsafe_allow_html=True)
