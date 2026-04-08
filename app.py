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

html, body {
    background: var(--bg-gradient) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
    font-size: 16px;
}

/* DASHBOARD CONTENT OFFSET & HIDE DEFAULTS */
header { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.main .block-container {
    padding-top: 80px !important;
    max-width: 95% !important;
}

/* CARDS & GRID */
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

/* GENERIC BUTTONS (REST OF APP) */
div.stButton > button:not([data-testid^="baseButton-nav_"]) {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}

div.stButton > button:not([data-testid^="baseButton-nav_"]):hover { 
    border-color: var(--accent-cyan) !important; 
    color: var(--accent-cyan) !important; 
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.2) !important; 
}

/* Hide Sidebar */
section[data-sticky-sidebar="true"] { display: none !important; }

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
if 'config' not in st.session_state:
    st.session_state.config = {
        "difficulty": "Medium",
        "agent_choice": "Baseline",
        "speed": 0.2,
        "seed": 42
    }
if 'env_initialized' not in st.session_state: st.session_state.env_initialized = False
if 'running' not in st.session_state: st.session_state.running = False
if 'paused' not in st.session_state: st.session_state.paused = False
if 'rewards' not in st.session_state: st.session_state.rewards = []
if 'log_history' not in st.session_state: st.session_state.log_history = []
if 'compare_results' not in st.session_state: st.session_state.compare_results = None

if 'env' not in st.session_state:
    # Pre-initialize a default environment for tactical preview
    st.session_state.env = CrisisEnv(difficulty=st.session_state.config["difficulty"].lower(), seed=st.session_state.config["seed"])
    st.session_state.state = st.session_state.env.reset()
    st.session_state.agent = BaselineAgent(st.session_state.env)
    st.session_state.prev_state = None

def render_navbar():
    active = st.session_state.get("active_page", "Scenario")
    status_color = "#00E5FF" if st.session_state.get("running") else "rgba(156,163,175,0.6)"
    status_text = "ACTIVE" if st.session_state.get("running") else "READY"

    def link_style(page):
        if active == page:
            return "font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#00E5FF;text-decoration:none;padding:7px 16px;border-radius:30px;background:rgba(0,229,255,0.12);border:1px solid rgba(0,229,255,0.35);white-space:nowrap;"
        return "font-family:'Orbitron',sans-serif;font-size:11px;font-weight:500;letter-spacing:1.5px;text-transform:uppercase;color:rgba(229,231,235,0.55);text-decoration:none;padding:7px 16px;border-radius:30px;white-space:nowrap;"

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(11,15,20,0.85);backdrop-filter:blur(20px);padding:6px 12px;border-radius:50px;border:1px solid rgba(255,255,255,0.08);margin:0 auto 30px auto;width:fit-content;gap:20px;box-shadow:0 12px 30px rgba(0,0,0,0.4);">
        <div style="font-family:'Orbitron',sans-serif;font-weight:800;font-size:14px;color:#00E5FF;letter-spacing:1px;padding-left:10px;">
            ⚡ CRISISFLOW
        </div>
        <div style="display:flex;gap:5px;background:rgba(0,0,0,0.2);padding:4px;border-radius:40px;border:1px solid rgba(255,255,255,0.03);">
            <a href="/?page=Scenario" style="{link_style('Scenario')}">Scenario</a>
            <a href="/?page=Simulation" style="{link_style('Simulation')}">Simulation</a>
            <a href="/?page=Compare" style="{link_style('Compare')}">Compare</a>
            <a href="/?page=Analytics" style="{link_style('Analytics')}">Analytics</a>
        </div>
        <div style="display:flex;align-items:center;gap:12px;padding-right:15px;border-left:1px solid rgba(255,255,255,0.08);padding-left:20px;">
            <div style="width:8px;height:8px;border-radius:50%;background:{status_color};box-shadow:0 0 10px {status_color};"></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(156,163,175,0.8);letter-spacing:1px;">
                ● STATUS: {status_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# ── PAGES ──
def render_scenario():
    st.markdown("<h1>Strategic Command Configuration</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.3, 0.45, 0.25])
    
    with c1:
        st.markdown('<div class="glass-card cyan-glow"><div class="card-title">Scenario Alignment</div>', unsafe_allow_html=True)
        # Use config for inputs
        diff = st.radio("Difficulty", ["Easy", "Medium", "Hard"], 
                        index=["Easy", "Medium", "Hard"].index(st.session_state.config["difficulty"]), horizontal=True)
        st.session_state.config["difficulty"] = diff
        
        agent = st.selectbox("Strategic Agent", ["Baseline", "Improved"], 
                             index=["Baseline", "Improved"].index(st.session_state.config["agent_choice"]))
        st.session_state.config["agent_choice"] = agent
        
        spd = st.slider("Response Speed", 1, 10, 5, step=1)
        st.session_state.config["speed"] = spd # Note: Using 1/speed for delay in sim
        
        if st.button("Initialize Environment", type="primary", width="stretch"):
            # True Initialization
            st.session_state.env = CrisisEnv(difficulty=st.session_state.config["difficulty"].lower(), 
                                            seed=st.session_state.config["seed"])
            st.session_state.state = st.session_state.env.reset()
            
            from crisisflow.agents.improved_agent import ImprovedAgent
            st.session_state.agent = (ImprovedAgent if st.session_state.config["agent_choice"] == "Improved" else BaselineAgent)(st.session_state.env)
            
            st.session_state.env_initialized = True
            st.session_state.active_page = "Simulation"
            st.session_state.rewards = []
            st.session_state.log_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card"><div class="card-title">Tactical Preview Map</div>', unsafe_allow_html=True)
        # Always show current state preview
        s_dict = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
        st.plotly_chart(draw_pydeck_map(s_dict), width="stretch", key="preview_map_chart")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c3:
        info = st.session_state.env._build_info()
        st.markdown(render_glass_card("Tactical Intelligence", f"{info['survival_rate']*100:.1f}%", "Expected Survival Probability"), unsafe_allow_html=True)
        st.markdown(f"""<div class="glass-card"><div class="card-title">System Status</div><div class="card-desc">Environment locked at <b>{st.session_state.config['difficulty']}</b>. Strategic cluster initialized with <b>{st.session_state.config['agent_choice']}</b> agent.</div></div>""", unsafe_allow_html=True)

def render_simulation():
    if not st.session_state.env_initialized:
        st.info("💡 Tactical environment not initialized. Go to **Scenario** to configure your mission.")
        return

    # ── CONTROLS ──
    st.markdown('<div class="glass-card" style="padding:10px 40px; margin-bottom:10px;">', unsafe_allow_html=True)
    ctrl = st.columns([1,1,1,1,8])
    with ctrl[0]:
        start_label = "▶ Resume" if st.session_state.paused else "▶ Start"
        if st.button(start_label, width="stretch"): 
            st.session_state.running = True
            st.session_state.paused = False
    with ctrl[1]:
        if st.button("⏸ Pause", width="stretch"): 
            st.session_state.paused = True
            st.session_state.running = False
    with ctrl[2]:
        if st.button("⏹ Stop", width="stretch"): 
            st.session_state.running = False
            st.session_state.paused = False
    with ctrl[3]:
        if st.button("🔁 Reset", width="stretch"):
            st.session_state.running = False
            st.session_state.paused = False
            st.session_state.state = st.session_state.env.reset()
            st.session_state.rewards = []; st.session_state.log_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── LAYOUT ──
    c_map, c_meta = st.columns([0.7, 0.3])
    with c_map:
        st.markdown('<div class="glass-card"><div class="card-title">Live Deployment Visualization</div>', unsafe_allow_html=True)
        map_p = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with c_meta:
        eff_p = st.empty()
        st.markdown('<div class="glass-card"><div class="card-title">🚨 Live Dispatch Log</div>', unsafe_allow_html=True)
        log_p = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card"><div class="card-title">📈 Reward Performance</div>', unsafe_allow_html=True)
        rew_p = st.empty()
        st.markdown('<div class="card-desc">Agent performance per timestep</div></div>', unsafe_allow_html=True)

    # ── SIMULATION LOOP ──
    if st.session_state.running:
        MAX_STEPS = 100
        while st.session_state.running and not st.session_state.paused:
            if st.session_state.active_page != "Simulation": 
                st.session_state.running = False
                break
            
            s = st.session_state.state
            act = st.session_state.agent.select_action(s)
            
            # Step the environment
            s2, r, d, inf = st.session_state.env.step(act)
            st.session_state.prev_state = s
            st.session_state.state = s2
            st.session_state.rewards.append(r.score if hasattr(r, "score") else r)
            
            # Update logs
            for a in act:
                st.session_state.log_history.insert(0, f'<div class="log-entry"><b class="log-entry-cyan">AMB-{a["ambulance_id"]:02d}</b> → <b class="log-entry-teal">Patient {a["patient_id"]:02d}</b></div>')
            if len(st.session_state.log_history) > 20: st.session_state.log_history = st.session_state.log_history[:20]

            # ── RENDER UPDATES (PLACEHOLDERS) ──
            sd = s2.model_dump() if hasattr(s2, "model_dump") else s2
            map_p.plotly_chart(draw_pydeck_map(sd, prev_state=st.session_state.prev_state, alpha=0.3), width="stretch", key="map_live")
            eff_p.markdown(render_glass_card("Mission Efficiency", f"{inf['score']*100:.1f}%", f"Step {len(st.session_state.rewards)}/{MAX_STEPS}", accent=True), unsafe_allow_html=True)
            log_p.markdown(f'<div class="dispatch-log">{"".join(st.session_state.log_history)}</div>', unsafe_allow_html=True)
            
            if st.session_state.rewards:
                fig = go.Figure(data=[go.Scatter(y=st.session_state.rewards, line=dict(color="#00E5FF", width=3), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)')])
                fig.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False)
                rew_p.plotly_chart(fig, width="stretch", config={'displayModeBar': False}, key="rew_live")
            
            if d or len(st.session_state.rewards) >= MAX_STEPS: 
                st.session_state.running = False
                st.rerun()
                break
            
            # Control speed: higher speed value = smaller sleep
            time.sleep(1.0 / st.session_state.config["speed"])

    # Post-Simulation / Paused View (Static)
    inf = st.session_state.env._build_info()
    eff_p.markdown(render_glass_card("Mission Efficiency", f"{inf['score']*100:.1f}%", "Mission Grade", accent=True), unsafe_allow_html=True)
    log_p.markdown(f'<div class="dispatch-log">{"".join(st.session_state.log_history)}</div>', unsafe_allow_html=True)
    sd = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
    map_p.plotly_chart(draw_pydeck_map(sd, alpha=1.0), width="stretch", key="map_static")
    if st.session_state.rewards:
        fig = go.Figure(data=[go.Scatter(y=st.session_state.rewards, line=dict(color="#00E5FF", width=3))])
        fig.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        rew_p.plotly_chart(fig, width="stretch", key="rew_static")

def render_comparison():
    st.markdown("<h1>Strategic Agent Comparison</h1>", unsafe_allow_html=True)
    if not st.session_state.compare_results:
        st.markdown(f"""<div class="glass-card" style="text-align:center;"><div class="card-title">Battle Benchmark</div><div class="card-desc">Execute simultaneous simulations to identify optimal response logic.</div></div>""", unsafe_allow_html=True)
        if st.button("Run Benchmark Analysis", type="primary"):
            with st.spinner("⚔️ Benchmarking..."):
                st.session_state.compare_results = get_benchmarking_results(st.session_state.config["difficulty"], 42)
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
            st.plotly_chart(fig, width='stretch', key="compare_bar_chart")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card"><div class="card-title">Learning Curve</div>', unsafe_allow_html=True)
            fig = go.Figure(); fig.add_trace(go.Scatter(y=b["reward_history"], name="Baseline", line=dict(color="#9CA3AF"))); fig.add_trace(go.Scatter(y=i["reward_history"], name="Improved", line=dict(color="#00E5FF", width=3)))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB', height=300)
            st.plotly_chart(fig, width='stretch', key="compare_line_chart")
            st.markdown('</div>', unsafe_allow_html=True)

def render_analytics():
    st.markdown("<h1>📊 Tactical Performance Analytics</h1>", unsafe_allow_html=True)
    if st.session_state.rewards:
        st.markdown('<div class="glass-card"><div class="card-title">Historical Reward Quality</div>', unsafe_allow_html=True)
        # Using a Plotly chart instead of st.line_chart for better styling integration
        fig = go.Figure(data=[go.Scatter(y=st.session_state.rewards, line=dict(color="#00E5FF", width=3), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)')])
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB')
        st.plotly_chart(fig, width='stretch', key="analytics_trend_chart")
        st.markdown('<div class="card-desc">Visualizing the real-time decision quality and strategic optimization of the chosen agent logic.</div></div>', unsafe_allow_html=True)
    else: 
        st.markdown('<div class="glass-card"><div class="card-title">Intelligence Gap</div><div class="card-desc">No mission data detected. Run a simulation to generate tactical analytics.</div></div>', unsafe_allow_html=True)


# ── MAIN ──
url_page = st.query_params.get("page", None)
if url_page in ["Scenario", "Simulation", "Compare", "Analytics"]:
    st.session_state.active_page = url_page
page = st.session_state.get("active_page", "Scenario")

render_navbar()

if page == "Scenario": render_scenario()
elif page == "Simulation": render_simulation()
elif page == "Compare": render_comparison()
elif page == "Analytics": render_analytics()
