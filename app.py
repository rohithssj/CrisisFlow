import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import random
import pydeck as pdk
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.agents.baseline_agent import BaselineAgent
from crisisflow.ui.map import draw_pydeck_map

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="CRISISFLOW | COMMAND CENTER",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PREMIUM GLASSMORPHISM DESIGN SYSTEM (MANDATORY) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0B0F14;
    --bg-gradient: linear-gradient(135deg, #0B0F14 0%, #111827 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-cyan: #00E5FF;
    --accent-green: #00FF88;
    --accent-warning: #FFA500;
    --accent-danger: #FF3B3B;
    --text-primary: #E5E7EB;
    --text-muted: #9CA3AF;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-gradient) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
}

/* Glass Card Component */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.cyan-glow {
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    border-color: rgba(0, 229, 255, 0.3) !important;
}

/* Typography Classes */
.title-bold { font-weight: 800; text-transform: uppercase; letter-spacing: 2px; }
.metric-large { font-size: 42px; font-weight: 800; line-height: 1; margin: 8px 0; }
.label-small { font-size: 12px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

/* Metrics Grid Helper */
.metrics-grid-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.metrics-grid-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--glass-border);
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}

/* Log Container */
.intelligence-feed {
    height: 250px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    background: rgba(0, 0, 0, 0.2);
    padding: 10px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.log-entry { margin-bottom: 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.03); padding-bottom: 2px; }
.log-cyan { color: var(--accent-cyan); }
.log-red { color: var(--accent-danger); }
.log-green { color: var(--accent-green); }

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Sidebar Overrides */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 20, 0.95) !important;
    border-right: 1px solid var(--glass-border);
}
/* Comparison Table */
.comp-table {
    width: 100%;
    margin-top: 10px;
    border-collapse: collapse;
}

.comp-table th, .comp-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--glass-border);
}

.comp-table th {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.comp-table td {
    font-size: 14px;
    font-weight: 600;
}

.winner-badge {
    background: linear-gradient(90deg, rgba(0, 229, 255, 0.2), transparent);
    border-left: 4px solid var(--accent-cyan);
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 20px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── LOGIC HELPER ──
def get_benchmarking_results(difficulty, seed):
    from crisisflow.agents.improved_agent import ImprovedAgent
    res = {}
    agents = [
        ("Baseline", BaselineAgent),
        ("Improved", ImprovedAgent)
    ]
    for name, agent_cls in agents:
        env = CrisisEnv(difficulty=difficulty.lower(), seed=seed)
        agent = agent_cls(env)
        s = env.reset()
        d = False
        reward_history = []
        while not d:
            a = agent.select_action(s)
            s, r, d, inf = env.step(a)
            # Handle reward object or scalar
            r_val = r.score if hasattr(r, 'score') else r
            reward_history.append(r_val)
        
        # Add historical metrics to info
        inf['reward_history'] = reward_history
        res[name] = inf
    return res

# ── SESSION STATE ──
if 'running' not in st.session_state: st.session_state.running = False
if 'rewards' not in st.session_state: st.session_state.rewards = []
if 'env' not in st.session_state: st.session_state.env = None
if 'state' not in st.session_state: 
    _init_env = CrisisEnv(difficulty="medium", seed=42)
    st.session_state.state = _init_env.reset()
    st.session_state.env = _init_env
    st.session_state.agent = BaselineAgent(_init_env)
    st.session_state.prev_state = None

if 'speed' not in st.session_state: st.session_state.speed = 0.3
if 'agent_type' not in st.session_state: st.session_state.agent_type = "Baseline"
if 'log_history' not in st.session_state: st.session_state.log_history = []
if 'compare_results' not in st.session_state: st.session_state.compare_results = None

# ── SIDEBAR: CONTROL PANEL ──
with st.sidebar:
    st.markdown("<h1 class='title-bold' style='color:var(--accent-cyan); margin-bottom:0;'>⚡ CrisisFlow</h1>", unsafe_allow_html=True)
    st.markdown("<p class='label-small' style='margin-top:-10px; margin-bottom:24px;'>Command Center</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='label-small'>1. Scenario Selector</p>", unsafe_allow_html=True)
    diff = st.radio("Difficulty", ["Easy", "Medium", "Hard"], horizontal=True, label_visibility="collapsed", index=1)
    
    st.markdown("<br><p class='label-small'>2. Agent Selector</p>", unsafe_allow_html=True)
    agent_choice = st.radio("Agent", ["Baseline", "Improved", "Compare Both"], horizontal=True, label_visibility="collapsed", index=0)
    st.session_state.agent_type = agent_choice
    
    st.markdown("<br><p class='label-small'>3. Simulation Controls</p>", unsafe_allow_html=True)
    sim_speed = st.slider("Response Speed", 0.05, 0.5, 0.2, step=0.05, label_visibility="collapsed")
    st.session_state.speed = sim_speed
    explain_toggle = st.toggle("Explain Decisions", value=True)
    
    st.markdown("<br><p class='label-small'>4. Actions</p>", unsafe_allow_html=True)
    if st.button("▶ Start Simulation", type="primary", use_container_width=True):
        from crisisflow.agents.improved_agent import ImprovedAgent
        
        if agent_choice == "Compare Both":
            with st.spinner("⚔️ Tactical Benchmarking in Progress..."):
                st.session_state.compare_results = get_benchmarking_results(diff, 42)
                st.session_state.running = False # Don't run the regular loop
                st.toast("COMPILATION COMPLETE", icon="📊")
        else:
            st.session_state.running = True
            st.session_state.env = CrisisEnv(difficulty=diff.lower(), seed=42)
            if agent_choice == "Baseline":
                st.session_state.agent = BaselineAgent(st.session_state.env)
            else:
                st.session_state.agent = ImprovedAgent(st.session_state.env)
            st.session_state.state = st.session_state.env.reset()
            st.session_state.rewards = []
            st.session_state.log_history = []
            st.session_state.compare_results = None
        
    if st.button("🔄 Reset Scenario", use_container_width=True):
        st.session_state.running = False
        st.session_state.rewards = []
        st.session_state.log_history = []
        st.rerun()

# ── COMPONENT HELPERS (GLASSMORPHISM) ──
def get_mission_efficiency_card(value):
    return f"""
    <div class="glass-card cyan-glow" style="text-align:center;">
        <div class="label-small">Mission Efficiency</div>
        <div class="metric-large" style="color:var(--accent-cyan); text-shadow: 0 0 15px rgba(0,229,255,0.4);">{value}</div>
    </div>
    """

def get_metrics_grid(survival, deaths, critical, response):
    return f"""
    <div class="metrics-grid-container">
        <div class="metrics-grid-item">
            <div class="label-small">Survival Rate</div>
            <div style="font-size:24px; font-weight:800; color:var(--accent-green);">{survival}</div>
        </div>
        <div class="metrics-grid-item">
            <div class="label-small">Fatalities</div>
            <div style="font-size:24px; font-weight:800; color:var(--accent-danger);">{deaths}</div>
        </div>
        <div class="metrics-grid-item">
            <div class="label-small">Critical Saved</div>
            <div style="font-size:24px; font-weight:800; color:var(--accent-cyan);">{critical}</div>
        </div>
        <div class="metrics-grid-item">
            <div class="label-small">Avg Response</div>
            <div style="font-size:24px; font-weight:800; color:var(--accent-cyan);">{response}</div>
        </div>
    </div>
    """

# ── COMPARISON DASHBOARD RENDERER ──
def render_comparison_dashboard(results):
    if not results: return
    
    b = results.get("Baseline", {})
    i = results.get("Improved", {})
    
    # Calculate performance delta
    b_score = b.get('score', 0)
    i_score = i.get('score', 0)
    perf_delta = ((i_score - b_score) / b_score * 100) if b_score > 0 else 0
    
    st.markdown(f"""
    <div class="winner-badge">
        🏆 Improved Agent performs better (+{perf_delta:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Table
    st.markdown(f"""
    <table class="comp-table">
        <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Improved</th>
        </tr>
        <tr>
            <td>Survival Rate</td>
            <td style="color:var(--text-muted);">{b.get('survival_rate', 0)*100:.1f}%</td>
            <td style="color:var(--accent-green);">{i.get('survival_rate', 0)*100:.1f}%</td>
        </tr>
        <tr>
            <td>Deaths</td>
            <td style="color:var(--text-muted);">{b.get('deaths', 0)}</td>
            <td style="color:var(--accent-danger);">{i.get('deaths', 0)}</td>
        </tr>
        <tr>
            <td>Response Time</td>
            <td style="color:var(--text-muted);">{b.get('avg_response_time', 0):.2f}s</td>
            <td style="color:var(--accent-cyan);">{i.get('avg_response_time', 0):.2f}s</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<p class='label-small'>Score Comparison</p>", unsafe_allow_html=True)
        fig_bar = go.Figure(data=[
            go.Bar(name='Baseline', x=['Baseline'], y=[b_score*100], marker_color='#9CA3AF'),
            go.Bar(name='Improved', x=['Improved'], y=[i_score*100], marker_color='#00E5FF')
        ])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E5E7EB',
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown("<p class='label-small'>Reward Over Time</p>", unsafe_allow_html=True)
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(y=b.get('reward_history', []), name='Baseline', line=dict(color='#9CA3AF', width=2)))
        fig_line.add_trace(go.Scatter(y=i.get('reward_history', []), name='Improved', line=dict(color='#00E5FF', width=3)))
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#E5E7EB',
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
# ── MAIN COMMAND VIEWPORT ──
if st.session_state.agent_type == "Compare Both" and st.session_state.compare_results:
    st.markdown("<p class='label-small' style='margin-bottom:24px;'>Strategic Agent Comparison Dashboard</p>", unsafe_allow_html=True)
    render_comparison_dashboard(st.session_state.compare_results)
else:
    col_sidebar, col_map, col_intel = st.columns([0.2, 0.55, 0.25])
    
    with col_map:
        st.markdown("<p class='label-small' style='margin-bottom:8px;'>Tactical Deployment Map</p>", unsafe_allow_html=True)
        map_placeholder = st.empty()
        
        # Grid Status Info
        info_cols = st.columns(3)
        with info_cols[0]: step_placeholder = st.empty()
        with info_cols[1]: status_placeholder = st.empty()
        with info_cols[2]: active_count_placeholder = st.empty()

    with col_intel:
        st.markdown("<p class='label-small' style='margin-bottom:8px;'>Intelligence System</p>", unsafe_allow_html=True)
        
        # 🟦 TOP CARD — MISSION EFFICIENCY
        efficiency_placeholder = st.empty()
        
        # 📦 METRICS GRID (2x2)
        metrics_placeholder = st.empty()
        
        # ⚔️ AGENT COMPARISON
        st.markdown("<br><p class='label-small'>Agent Comparison</p>", unsafe_allow_html=True)
        comparison_placeholder = st.empty()
        
        # 🧠 OPERATIONAL INTELLIGENCE FEED
        st.markdown("<br><p class='label-small'>Operational Intelligence Feed</p>", unsafe_allow_html=True)
        log_placeholder = st.empty()
        
        # 📈 REWARD PERFORMANCE CHART
        # 📈 REWARD PERFORMANCE CHART
        st.markdown("<br><p class='label-small'>Reward Performance</p>", unsafe_allow_html=True)
        chart_placeholder = st.empty()

    # ── TACTICAL VIEW LOGIC (Only runs when placeholders exist) ──
    def update_ui_placeholders(info, rewards, logs, compare_data=None):
        efficiency_placeholder.markdown(get_mission_efficiency_card(f"{info['score']*100:.1f}%"), unsafe_allow_html=True)
        metrics_placeholder.markdown(get_metrics_grid(
            f"{info['survival_rate']*100:.1f}%",
            f"{info['deaths']}",
            f"{info['critical_saved']}",
            f"{info['avg_response_time']}s"
        ), unsafe_allow_html=True)
        
        # Logs
        log_content = "".join(logs)
        log_placeholder.markdown(f'<div class="intelligence-feed">{log_content}</div>', unsafe_allow_html=True)
        
        # Chart
        if rewards:
            chart_placeholder.line_chart(rewards, color="#00E5FF")
        
        # Comparison Section
        # (Removed old inline comparison logic in favor of dedicated dashboard)
        pass

    # Initial Render
    if st.session_state.state:
        env_info = st.session_state.env._build_info() if st.session_state.env else {"score": 0, "survival_rate": 0, "deaths": 0, "critical_saved": 0, "avg_response_time": 0}
        update_ui_placeholders(env_info, st.session_state.rewards, st.session_state.log_history)
        
        state_dict = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
        fig = draw_pydeck_map(state_dict, prev_state=st.session_state.prev_state, alpha=1.0)
        # map_placeholder.plotly_chart is called in the loop or initial render
        # Initial map render:
        map_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Simulation Inner Loop
    if st.session_state.running:
        env = st.session_state.env
        agent = st.session_state.agent
        
        while st.session_state.running:
            state = st.session_state.state
            state_dict = state.model_dump() if hasattr(state, "model_dump") else state
            
            # Select Action
            actions = agent.select_action(state)
            
            # Decision Logging
            for action in actions:
                p_id = action['patient_id']
                a_id = action['ambulance_id']
                p_data = next((p for p in state_dict['patients'] if p['id'] == p_id), None)
                sev = p_data['severity'] if p_data else 1
                sev_class = ["log-green", "log-cyan", "log-red"][sev-1]
                st.session_state.log_history.insert(0, f'<div class="log-entry">AMB-{a_id:02d} <span class="log-cyan">→</span> <span class="{sev_class}">P{p_id:02d}</span></div>')
                if len(st.session_state.log_history) > 50: st.session_state.log_history.pop()

            # Step Environment
            next_state, reward, done, info = env.step(actions)
            
            # State Update
            st.session_state.prev_state = state
            st.session_state.state = next_state
            
            reward_val = reward.score if hasattr(reward, 'score') else reward
            st.session_state.rewards.append(reward_val)
            
            # ── UPDATE UI ──
            update_ui_placeholders(info, st.session_state.rewards, st.session_state.log_history)
            
            # Map Update
            next_state_dict = next_state.model_dump() if hasattr(next_state, "model_dump") else next_state
            fig = draw_pydeck_map(next_state_dict, prev_state=st.session_state.prev_state, alpha=0.3)
            map_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Grid Status Update
            step_placeholder.markdown(f"<div class='label-small'>Step</div><div style='font-size:18px; font-weight:800;'>{next_state_dict['time_step']}</div>", unsafe_allow_html=True)
            status_placeholder.markdown(f"<div class='label-small'>Status</div><div style='font-size:18px; font-weight:800; color:var(--accent-green);'>ACTIVE</div>", unsafe_allow_html=True)
            active_count = sum(1 for p in next_state_dict['patients'] if not p['rescued'] and not p['dead'])
            active_count_placeholder.markdown(f"<div class='label-small'>Active Crises</div><div style='font-size:18px; font-weight:800; color:var(--accent-warning);'>{active_count}</div>", unsafe_allow_html=True)

            if done:
                st.session_state.running = False
                st.toast("MISSION COMPLETE", icon="✅")
                break
            
            time.sleep(st.session_state.speed)
