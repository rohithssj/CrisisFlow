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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PREMIUM COMMAND CENTER DESIGN SYSTEM ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0B0F14;
    --bg-gradient: linear-gradient(180deg, #0B0F14 0%, #05070A 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-cyan: #00E5FF;
    --accent-teal: #00FFD1;
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
    font-size: 16px;
}

/* FIXED TOP NAVBAR */
.nav-bar-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 70px;
    background: rgba(11, 15, 20, 0.95);
    backdrop-filter: blur(25px);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 40px;
    z-index: 99999;
}

.nav-logo {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 24px;
    letter-spacing: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
}

.nav-item { position: relative; display: flex; justify-content: center; }
.nav-item::after, .nav-item::before, .nav-item.active::after { display: none !important; content: none !important; border: none !important; }

/* GLASS CARDS */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(20px);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    opacity: 0.7;
}

.card-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 32px;
    font-weight: bold;
    color: var(--accent-cyan);
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
    margin-bottom: 6px;
}

.card-desc { font-size: 12px; color: var(--text-muted); opacity: 0.6; }
.cyan-glow { border-color: rgba(0, 229, 255, 0.4) !important; background: rgba(0, 229, 255, 0.05); }

/* TYPOGRAPHY */
h1 { font-family: 'Orbitron', sans-serif !important; font-size: 24px !important; color: var(--accent-cyan); margin-bottom: 24px !important; }

/* BUTTONS */
div.stButton > button {
    background: transparent !important;
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 20px;
}

div.stButton > button:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); box-shadow: 0 0 10px rgba(0, 229, 255, 0.1); }
div.stButton > button:focus { outline: none !important; box-shadow: none !important; }

[data-testid="stAppViewBlockContainer"] { padding-top: 100px !important; max-width: 95% !important; }
[data-testid="stSidebar"] { display: none !important; }

/* FEED LOGS */
.dispatch-log {
    height: 350px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    background: rgba(0, 0, 0, 0.4);
    padding: 12px;
    border-radius: 8px;
}

.log-entry { margin-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.03); padding-bottom: 4px; }
.log-entry-cyan { color: var(--accent-cyan); }
.log-entry-teal { color: var(--accent-teal); }

/* HIDE DEFAULTS */
#MainMenu, footer, .stDeployButton { visibility: hidden; display: none !important; }
[data-testid="stHeader"] { background: transparent !important; height: 0; }
</style>
""", unsafe_allow_html=True)

# ── LOGIC HELPERS ──
def get_benchmarking_results(difficulty, seed):
    from crisisflow.agents.improved_agent import ImprovedAgent
    res = {}
    for name, agent_cls in [("Baseline", BaselineAgent), ("Improved", ImprovedAgent)]:
        env = CrisisEnv(difficulty=difficulty.lower(), seed=seed)
        agent = agent_cls(env)
        s = env.reset(); d = False; r_hist = []
        while not d:
            a = agent.select_action(s)
            s, r, d, inf = env.step(a)
            r_hist.append(r.score if hasattr(r, 'score') else r)
        inf['reward_history'] = r_hist
        res[name] = inf
    return res

def render_glass_card(title, value, desc="", accent=False):
    accent_class = "cyan-glow" if accent else ""
    return f"""
    <div class="glass-card {accent_class}">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-desc">{desc}</div>
    </div>
    """

# ── SESSION STATE ──
if 'active_page' not in st.session_state: st.session_state.active_page = "Scenario"
if 'difficulty' not in st.session_state: st.session_state.difficulty = "Medium"
if 'agent_choice' not in st.session_state: st.session_state.agent_choice = "Baseline"
if 'speed' not in st.session_state: st.session_state.speed = 0.2
if 'running' not in st.session_state: st.session_state.running = False
if 'paused' not in st.session_state: st.session_state.paused = False
if 'rewards' not in st.session_state: st.session_state.rewards = []
if 'log_history' not in st.session_state: st.session_state.log_history = []
if 'compare_results' not in st.session_state: st.session_state.compare_results = None
if 'env' not in st.session_state:
    st.session_state.env = CrisisEnv(difficulty="medium", seed=42)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.agent = BaselineAgent(st.session_state.env)
    st.session_state.prev_state = None

# ── NAVIGATION ──
def render_navbar():
    st.markdown('<div class="nav-bar-container"></div>', unsafe_allow_html=True)
    st.markdown('<div style="position:fixed; left:48px; top:18px; z-index:100001;" class="nav-logo">CrisisFlow ⚡</div>', unsafe_allow_html=True)
    cols = st.columns([0.4, 0.15, 0.15, 0.15, 0.15])
    pages = ["Scenario", "Simulation", "Compare", "Analytics"]
    for i, page in enumerate(pages):
        with cols[i+1]:
            if st.button(page, key=f"nav_{page}"):
                st.session_state.active_page = page
                st.rerun()

# ── PAGES ──
def render_scenario():
    st.markdown("<h1>Strategic Command Configuration</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.3, 0.45, 0.25])
    with c1:
        st.markdown('<div class="glass-card cyan-glow"><div class="card-title">Scenario Alignment</div>', unsafe_allow_html=True)
        st.session_state.difficulty = st.radio("Difficulty", ["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty), horizontal=True)
        st.session_state.agent_choice = st.selectbox("Strategic Agent", ["Baseline", "Improved"], index=["Baseline", "Improved"].index(st.session_state.agent_choice))
        st.session_state.speed = st.slider("Response Speed", 0.05, 0.5, 0.2, step=0.05)
        if st.button("Initialize Environment", type="primary", use_container_width=True):
            st.session_state.env = CrisisEnv(difficulty=st.session_state.difficulty.lower())
            st.session_state.state = st.session_state.env.reset()
            from crisisflow.agents.improved_agent import ImprovedAgent
            st.session_state.agent = (ImprovedAgent if st.session_state.agent_choice == "Improved" else BaselineAgent)(st.session_state.env)
            st.session_state.active_page = "Simulation"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><div class="card-title">Tactical Preview Map</div>', unsafe_allow_html=True)
        s_dict = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
        st.plotly_chart(draw_pydeck_map(s_dict), use_container_width=True, key="preview_map_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        info = st.session_state.env._build_info()
        st.markdown(render_glass_card("Tactical Intelligence", f"{info['survival_rate']*100:.1f}%", "Expected Survival Probability"), unsafe_allow_html=True)
        st.markdown(f"""<div class="glass-card"><div class="card-title">System Status</div><div class="card-desc">Environment locked at <b>{st.session_state.difficulty}</b>. Strategic cluster initialized with <b>{st.session_state.agent_choice}</b> agent.</div></div>""", unsafe_allow_html=True)

def render_simulation():
    # ── CONTROLS ──
    st.markdown('<div class="glass-card" style="padding:10px 40px; margin-bottom:10px;">', unsafe_allow_html=True)
    ctrl = st.columns([1,1,1,1,8])
    with ctrl[0]:
        if st.button("▶ Start", use_container_width=True): st.session_state.running = True
    with ctrl[1]:
        if st.button("⏸ Pause", use_container_width=True): st.session_state.paused = not st.session_state.paused
    with ctrl[2]:
        if st.button("⏹ Stop", use_container_width=True): st.session_state.running = False
    with ctrl[3]:
        if st.button("🔁 Reset", use_container_width=True):
            st.session_state.running = False
            st.session_state.state = st.session_state.env.reset()
            st.session_state.rewards = []; st.session_state.log_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── LAYOUT ──
    c_map, c_meta = st.columns([0.7, 0.3])
    
    # Simulation Logic - ELIMINATED WHILE LOOP TO AVOID DUPLICATE ID/KEY ERRORS
    # Every tick of the simulation is now one script execution (st.rerun pattern)
    if st.session_state.running and not st.session_state.paused:
        s = st.session_state.state
        act = st.session_state.agent.select_action(s)
        for a in act:
            st.session_state.log_history.insert(0, f'<div class="log-entry"><b class="log-entry-cyan">AMB-{a["ambulance_id"]:02d}</b> → <b class="log-entry-teal">Patient {a["patient_id"]:02d}</b></div>')
        if len(st.session_state.log_history) > 30: st.session_state.log_history = st.session_state.log_history[:30]

        s2, r, d, inf = st.session_state.env.step(act)
        st.session_state.prev_state = s
        st.session_state.state = s2
        st.session_state.rewards.append(r.score if hasattr(r, "score") else r)
        
        if d:
            st.session_state.running = False
            st.toast("MISSION COMPLETE")

    # ── RENDER MAP ──
    with c_map:
        st.markdown('<div class="glass-card"><div class="card-title">Live Deployment Visualization</div>', unsafe_allow_html=True)
        sd = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
        # Static keys are now SAFE because there is NO while loop in a single run
        st.plotly_chart(draw_pydeck_map(sd, prev_state=st.session_state.prev_state, alpha=0.3), use_container_width=True, key="sim_map_live")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RENDER METRICS & LOGS ──
    with c_meta:
        inf = st.session_state.env._build_info()
        st.markdown(render_glass_card("Mission Efficiency", f"{inf['score']*100:.1f}%", "Total Mission Grade", accent=True), unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card"><div class="card-title">🚨 Live Dispatch Log</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dispatch-log">{"".join(st.session_state.log_history)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Reward Chart inside Card
        st.markdown('<div class="glass-card"><div class="card-title">📈 Reward Performance</div>', unsafe_allow_html=True)
        if st.session_state.rewards:
            fig = go.Figure(data=[go.Scatter(y=st.session_state.rewards, line=dict(color="#00E5FF", width=3), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)')])
            fig.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="sim_reward_live")
        st.markdown('<div class="card-desc">Agent performance per timestep</div></div>', unsafe_allow_html=True)

    # ── TRIGGER NEXT STEP ──
    if st.session_state.running and not st.session_state.paused:
        time.sleep(st.session_state.speed)
        st.rerun()

def render_comparison():
    st.markdown("<h1>Strategic Agent Comparison</h1>", unsafe_allow_html=True)
    if not st.session_state.compare_results:
        st.markdown(f"""<div class="glass-card" style="text-align:center;"><div class="card-title">Battle Benchmark</div><div class="card-desc">Execute simultaneous simulations to identify optimal response logic.</div></div>""", unsafe_allow_html=True)
        if st.button("Run Benchmark Analysis", type="primary"):
            with st.spinner("⚔️ Benchmarking..."):
                st.session_state.compare_results = get_benchmarking_results(st.session_state.difficulty, 42)
                st.rerun()
    else:
        res = st.session_state.compare_results; b, i = res["Baseline"], res["Improved"]
        diff = ((i["score"]-b["score"])/b["score"]*100) if b["score"] > 0 else 0
        st.markdown(render_glass_card("Command Verdict", f"Improved Agent: {diff:+.1f}%", "Superior performance logic detected in Improved Agent cluster.", accent=True), unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><div class="card-title">Performance Grade</div>', unsafe_allow_html=True)
            fig = go.Figure(data=[go.Bar(x=["Baseline", "Improved"], y=[b["score"]*100, i["score"]*100], marker_color=["#9CA3AF", "#00E5FF"])])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB', height=300)
            st.plotly_chart(fig, use_container_width=True, key="compare_bar_chart")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card"><div class="card-title">Learning Curve</div>', unsafe_allow_html=True)
            fig = go.Figure(); fig.add_trace(go.Scatter(y=b["reward_history"], name="Baseline", line=dict(color="#9CA3AF"))); fig.add_trace(go.Scatter(y=i["reward_history"], name="Improved", line=dict(color="#00E5FF", width=3)))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB', height=300)
            st.plotly_chart(fig, use_container_width=True, key="compare_line_chart")
            st.markdown('</div>', unsafe_allow_html=True)

def render_analytics():
    st.markdown("<h1>📊 Reward Trend Analysis</h1>", unsafe_allow_html=True)
    if st.session_state.rewards:
        st.markdown('<div class="glass-card"><div class="card-title">Historical Quality Progression</div>', unsafe_allow_html=True)
        st.line_chart(st.session_state.rewards)
        # st.line_chart doesn't take keys in same way, so it is safe.
        st.markdown('<div class="card-desc">Shows how decision quality evolves over time per timestep.</div></div>', unsafe_allow_html=True)
    else: st.warning("No mission data analyzed yet.")

# ── MAIN ──
render_navbar()
if st.session_state.active_page == "Scenario": render_scenario()
elif st.session_state.active_page == "Simulation": render_simulation()
elif st.session_state.active_page == "Compare": render_comparison()
elif st.session_state.active_page == "Analytics": render_analytics()
