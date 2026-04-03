import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.agents.baseline_agent import BaselineAgent

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="CrisisFlow AI Simulator",
    layout="wide"
)

# ── CUSTOM STYLING ──
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    .big-metric-value {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .big-metric-label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #888888;
        text-align: center;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# ── HELPERS ──

def draw_map(state):
    """
    Draws the city grid with professional visual encoding.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    
    # Grid
    ax.grid(color='#2a2d3e', linestyle='-', linewidth=0.5, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Traffic Zones (Subtle background)
    ax.axvspan(0.00, 0.33, color='#50fa7b', alpha=0.02, label="Low Traffic")
    ax.axvspan(0.33, 0.66, color='#f1fa8c', alpha=0.02, label="Medium Traffic")
    ax.axvspan(0.66, 1.00, color='#ff5555', alpha=0.02, label="High Traffic")
    
    # Hospitals (Biggest Markers)
    for h in state.get("hospitals", []):
        ax.plot(h['x'], h['y'], marker='s', color='#1a1c24', markersize=22, markeredgecolor='#4488ff', markeredgewidth=2)
        ax.text(h['x'], h['y'], 'H', color='#4488ff', weight='bold', ha='center', va='center', fontsize=12)
        ax.text(h['x'], h['y'] - 0.05, f"{h['current_load']}/{h['capacity']}", color='white', 
                fontsize=9, ha='center', fontweight='bold')

    # Patients (Medium-Large Circles)
    for p in state.get("patients", []):
        if p['dead']:
            ax.plot(p['x'], p['y'], 'x', color='#6272a4', markersize=10, markeredgewidth=2)
        elif p['rescued']:
            ax.plot(p['x'], p['y'], 'x', color='#f8f8f2', markersize=10, markeredgewidth=1)
        else:
            color = {1: '#50fa7b', 2: '#ffb86c', 3: '#ff5555'}[p['severity']]
            size = {1: 10, 2: 12, 3: 15}[p['severity']]
            ax.plot(p['x'], p['y'], 'o', color=color, markersize=size, markeredgecolor='black', markeredgewidth=0.5)

    # Ambulances (Large Blue Squares)
    for a in state.get("ambulances", []):
        color = '#f8f8f2'
        if a['busy']:
            color = '#4488ff'
            # Draw professional dashed dispatch line
            if a.get('target_patient_id') is not None:
                pat = next((p for p in state['patients'] if p['id'] == a['target_patient_id']), None)
                if pat:
                    ax.plot([a['x'], pat['x']], [a['y'], pat['y']], color='#4488ff', linestyle='--', alpha=0.4, linewidth=1.5)
        elif a.get('cooldown_remaining', 0) > 0:
            color = '#6272a4' # Cooldown gray
            
        ax.plot(a['x'], a['y'], 's', color=color, markersize=14, markeredgecolor='black', markeredgewidth=0.5)

    # Title
    plt.title("CrisisFlow Deployment Map", color='white', fontsize=16, weight='bold', pad=15)

    # Legend (Professional encoded legend)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#ff5555', label='Critical', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='#ffb86c', label='Serious', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='#50fa7b', label='Minor', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='x', color='#6272a4', label='Dead', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='s', color='#4488ff', label='Ambulance', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#1a1c24', label='Hospital', markersize=12, linestyle='None', 
               markeredgecolor='#4488ff', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small', framealpha=0.8, 
              facecolor='#1a1c24', labelcolor='white', title="Map Legend", title_fontsize='medium')

    return fig

def random_agent_action(state):
    """
    Implementation of RandomAgent for comparison.
    """
    active = [p for p in state["patients"] if not p["rescued"] and not p["dead"]]
    free = [a for a in state["ambulances"] if not a["busy"] and a.get("cooldown_remaining", 0) == 0]
    
    if not active or not free:
        return []
        
    actions = []
    used_pats = set()
    shuffled_free = free[:]
    random.shuffle(shuffled_free)
    
    for amb in shuffled_free:
        available = [p for p in active if p["id"] not in used_pats]
        if not available:
            break
        pat = random.choice(available)
        actions.append({"ambulance_id": amb["id"], "patient_id": pat["id"]})
        used_pats.add(pat["id"])
    return actions

# ── SESSION STATE ──
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'state' not in st.session_state:
    st.session_state.state = None
if 'info' not in st.session_state:
    st.session_state.info = None
if 'rewards' not in st.session_state:
    st.session_state.rewards = []
if 'running' not in st.session_state:
    st.session_state.running = False

# ── UI TABS ──
tab1, tab2 = st.tabs(["Simulation", "Agent Comparison"])

# ── TAB 1: SIMULATION ──
with tab1:
    # Sidebar
    st.sidebar.header("Controls")
    diff = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=99999, value=42)
    speed_level = st.sidebar.slider("Simulation Speed", 1, 5, value=3)
    
    speed_map = {1: 1.0, 2: 0.6, 3: 0.3, 4: 0.15, 5: 0.05}
    sleep_time = speed_map[speed_level]

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_clicked = col_btn1.button("Start Simulation")
    reset_clicked = col_btn2.button("Reset Simulation")

    if reset_clicked:
        st.session_state.running = False
        st.session_state.env = None
        st.session_state.state = None
        st.session_state.rewards = []
        st.rerun()

    # Layout columns
    left_col, right_col = st.columns([0.6, 0.4])

    # Right column (Metrics)
    with right_col:
        st.markdown('<p class="big-metric-label">Final Score</p>', unsafe_allow_html=True)
        big_score_placeholder = st.empty()
        
        st.markdown("### Performance Metrics")
        
        m_row1_col1, m_row1_col2 = st.columns(2)
        metric_survival = m_row1_col1.empty()
        metric_dead = m_row1_col2.empty()
        
        m_row2_col1, m_row2_col2 = st.columns(2)
        metric_critical = m_row2_col1.empty()
        metric_avg_resp = m_row2_col2.empty()
        # Fixed layout for row 2
        
        st.write("---")
        st.write("**Reward Per Step**")
        reward_chart = st.empty()
        
        st.write("---")
        st.write("**Hospital Infrastructure Status**")
        hosp_status = st.empty()
        
        step_caption = st.empty()

    # Left column (Map)
    with left_col:
        st.markdown("### Simulation Map")
        map_placeholder = st.empty()

    # Simulation Logic
    if start_clicked:
        st.session_state.running = True
        st.session_state.env = CrisisEnv(difficulty=diff.lower(), seed=seed)
        st.session_state.agent = BaselineAgent(st.session_state.env)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.rewards = []
        
        max_steps = {
            "easy": 500,
            "medium": 800,
            "hard": 1000
        }[diff.lower()]

        prev_rescued = 0
        
        while st.session_state.running:
            state = st.session_state.state
            env = st.session_state.env
            agent = st.session_state.agent
            
            actions = agent.select_action(state)
            next_state, reward, done, info = env.step(actions)
            
            st.session_state.state = next_state
            st.session_state.info = info
            st.session_state.rewards.append(reward)
            
            # Update UI
            # 1. Map
            fig = draw_map(next_state)
            map_placeholder.pyplot(fig)
            plt.close(fig)
            
            # 2. Metrics
            big_score_placeholder.markdown(f'<p class="big-metric-value">{info["score"]:.4f}</p>', unsafe_allow_html=True)
            
            metric_survival.metric("Survival Rate", f"{info['survival_rate']}%")
            
            metric_dead.metric("Total Deaths", info['dead'], delta=None, delta_color="inverse")
            
            # Using independent columns for second row of metrics
            metric_critical.metric("Critical Patients Saved", info['critical_rescued'])
            metric_avg_resp.metric("Average Response Time", f"{info['avg_response_time']} Steps")
            
            reward_chart.line_chart(st.session_state.rewards)
            
            # Hospital Infrastructure
            with hosp_status.container():
                for h in next_state['hospitals']:
                    st.progress(h['current_load'] / h['capacity'])
                    st.caption(f"Hospital {h['id']}: {h['current_load']}/{h['capacity']}")
            
            step_caption.caption(f"Step {next_state['step']} / {max_steps}")

            if done:
                st.session_state.running = False
                st.success(f"Episode Complete! Final Score: {info['score']:.4f} | Survival Rate: {info['survival_rate']}%")
                break
            
            time.sleep(sleep_time)

# ── TAB 2: AGENT COMPARISON ──
with tab2:
    st.header("Agent Comparison Dashboard")
    st.write("Compare the baseline agent's performance against a random-action agent on the same scenario.")
    
    comp_diff = st.selectbox("Comparison Difficulty", ["Easy", "Medium", "Hard"], index=1, key="comp_diff")
    comp_seed = st.number_input("Comparison Seed", min_value=0, max_value=99999, value=42, key="comp_seed")
    
    run_comp = st.button("Run Comparison")
    
    if run_comp:
        with st.spinner("Running simulations..."):
            results = {}
            
            # 1. Baseline Agent
            env_b = CrisisEnv(difficulty=comp_diff.lower(), seed=comp_seed)
            agent_b = BaselineAgent(env_b)
            state_b = env_b.reset()
            done_b = False
            while not done_b:
                acts = agent_b.select_action(state_b)
                state_b, _, done_b, info_b = env_b.step(acts)
            results['Baseline'] = info_b
            
            # 2. Random Agent
            env_r = CrisisEnv(difficulty=comp_diff.lower(), seed=comp_seed)
            state_r = env_r.reset()
            done_r = False
            while not done_r:
                acts = random_agent_action(state_r)
                state_r, _, done_r, info_r = env_r.step(acts)
            results['Random'] = info_r
            
            # Display Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Baseline Agent")
                st.metric("Final Score", f"{results['Baseline']['score']:.4f}")
                st.metric("Survival Rate", f"{results['Baseline']['survival_rate']}%")
                st.metric("Total Deaths", results['Baseline']['dead'])
                st.metric("Critical Patients Saved", results['Baseline']['critical_rescued'])
                st.metric("Average Response Time", f"{results['Baseline']['avg_response_time']} Steps")
            
            with col2:
                st.markdown("### Random Agent")
                st.metric("Final Score", f"{results['Random']['score']:.4f}")
                st.metric("Survival Rate", f"{results['Random']['survival_rate']}%")
                st.metric("Total Deaths", results['Random']['dead'])
                st.metric("Critical Patients Saved", results['Random']['critical_rescued'])
                st.metric("Average Response Time", f"{results['Random']['avg_response_time']} Steps")
            
            # Bar Chart Comparison
            import pandas as pd
            comp_data = pd.DataFrame({
                "Metric": ["Score", "Survival Rate", "Critical Saved"],
                "Baseline": [results['Baseline']['score'] * 100, results['Baseline']['survival_rate'], results['Baseline']['critical_rescued']],
                "Random": [results['Random']['score'] * 100, results['Random']['survival_rate'], results['Random']['critical_rescued']]
            }).set_index("Metric")
            
            st.write("---")
            st.write("📊 Performance Visual Comparison")
            st.bar_chart(comp_data)
            st.caption("Note: Score is multiplied by 100 for visual consistency with percentage metrics.")
            
            # Winner announcement
            if results['Baseline']['score'] > results['Random']['score']:
                st.success(f"🏆 Baseline Agent Wins by {results['Baseline']['score'] - results['Random']['score']:.4f} pts!")
            elif results['Baseline']['score'] < results['Random']['score']:
                st.error(f"⚠️ Random Agent Wins by {results['Random']['score'] - results['Baseline']['score']:.4f} pts!")
            else:
                st.info("🤝 It's a Tie!")
