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
    page_title="CRISISFLOW | AI COMMAND CENTER",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM DESIGN SYSTEM (CSS) ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Roboto+Mono:wght@400;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    :root {
        --bg-dark: #080a0f;
        --panel-bg: rgba(16, 22, 32, 0.8);
        --accent-blue: #00d4ff;
        --accent-red: #ff3e3e;
        --accent-orange: #ffa500;
        --accent-green: #00ff88;
        --text-main: #e0e6ed;
        --text-dim: #94a3b8;
        --border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main {
        background-color: var(--bg-dark);
        color: var(--text-main);
    }

    /* Force Dark Theme on App Container */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-dark) !important;
    }

    /* Metric Card Styling */
    .metric-card {
        background: var(--panel-bg);
        border: var(--border);
        border-radius: 12px;
        padding: 15px 10px; 
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .metric-label {
        color: var(--text-dim);
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
        display: block;
    }
    .metric-value {
        color: var(--accent-green);
        font-size: 28px;
        font-weight: 800;
        font-family: 'Roboto Mono', monospace;
        line-height: 1;
    }
    .final-score {
        background: linear-gradient(135deg, #111827 0%, #0c111a 100%);
        border: 2px solid var(--accent-blue);
        text-align: center;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
    }
    .final-score-label {
        font-size: 16px;
        color: var(--text-dim);
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .final-score-value {
        font-size: 48px;
        font-weight: 950;
        color: var(--accent-blue);
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin: 0;
        line-height: 1;
    }

    /* Decision Log Styling */
    .log-container {
        background: #111827;
        border: var(--border);
        border-radius: 10px;
        padding: 15px;
        height: 300px;
        overflow-y: auto;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-dim);
        margin-bottom: 20px;
    }
    .log-entry {
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .log-ambulance { color: var(--accent-blue); font-weight: bold; }
    .log-severity-3 { color: var(--accent-red); }
    </style>
    """, unsafe_allow_html=True)

# ── LOGIC HELPER ──
def get_benchmarking_results(difficulty, seed):
    res = {}
    for name, agent_func in [("Baseline", lambda e: BaselineAgent(e).select_action), ("Random", lambda e: random_agent_action)]:
        env = CrisisEnv(difficulty=difficulty.lower(), seed=seed)
        sel_act = agent_func(env)
        s = env.reset()
        d = False
        while not d:
            a = sel_act(s)
            s, _, d, inf = env.step(a)
        res[name] = inf
    return res

def random_agent_action(state):
    active = [p for p in state["patients"] if not p["rescued"] and not p["dead"]]
    free = [a for a in state["ambulances"] if not a["busy"] and not a.get("on_cooldown", False)]
    if not active or not free: return []
    actions = []
    used_pats = set()
    shuffled_free = free[:]
    random.shuffle(shuffled_free)
    for amb in shuffled_free:
        available = [p for p in active if p["id"] not in used_pats]
        if not available: break
        pat = random.choice(available)
        actions.append({"ambulance_id": amb["id"], "patient_id": pat["id"]})
        used_pats.add(pat["id"])
    return actions

# ── SESSION STATE ──
if 'running' not in st.session_state: st.session_state.running = False
if 'paused' not in st.session_state: st.session_state.paused = False
if 'rewards' not in st.session_state: st.session_state.rewards = []
if 'env' not in st.session_state: st.session_state.env = None
if 'state' not in st.session_state: 
    # Initialize a default state for the initial dashboard view
    _init_env = CrisisEnv(difficulty="medium", seed=42)
    st.session_state.state = _init_env.reset()
    st.session_state.env = _init_env
    st.session_state.agent = BaselineAgent(_init_env)

if 'speed' not in st.session_state: st.session_state.speed = 0.3

# ── SIDEBAR: NAVIGATION & CONTROLS ──
st.sidebar.markdown("<h1 style='color:#00d4ff; font-family:Roboto Mono;'>CRISISFLOW</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

nav = st.sidebar.radio("Command Navigation", ["Tactical Map", "Agent Comparison", "Mission Logs"])

st.sidebar.markdown("### Mission Configuration")
diff = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=1)
seed = st.sidebar.number_input("System Seed", min_value=0, value=42)

st.sidebar.markdown("### Simulation Speed")
s_col1, s_col2, s_col3 = st.sidebar.columns(3)
if s_col1.button("1x"): st.session_state.speed = 0.4
if s_col2.button("2x"): st.session_state.speed = 0.15
if s_col3.button("4x"): st.session_state.speed = 0.05

st.sidebar.markdown("### Execution")
c1, c2 = st.sidebar.columns(2)
if c1.button("Start Deployment", type="primary"):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.env = CrisisEnv(difficulty=diff.lower(), seed=seed)
    st.session_state.agent = BaselineAgent(st.session_state.env)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.rewards = []

if c2.button("Reset System"):
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.rewards = []
    st.rerun()

# ── COMPONENT HELPERS ──
def metric_card(title, value, color="#00ff88"):
    return f"""
    <div style="
        background:#111827;
        padding:16px;
        border-radius:12px;
        text-align:center;
        margin-bottom:10px;
        border:1px solid #1F2937;
    ">
        <div style="font-size:14px; color:#9CA3AF; margin-bottom:6px; text-transform:uppercase; font-weight:600; letter-spacing:1px;">
            {title}
        </div>
        <div style="font-size:28px; font-weight:700; color:{color}; font-family:'Roboto Mono', monospace;">
            {value}
        </div>
    </div>
    """

def big_score_card(value):
    return f"""
    <div style="
        background:#111827;
        padding:20px;
        border-radius:14px;
        text-align:center;
        border:1px solid #1F2937;
        margin-bottom:12px;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);
    ">
        <div style="font-size:14px; color:#9CA3AF; font-weight:700; letter-spacing:2px; margin-bottom:8px;">
            MISSION EFFICIENCY
        </div>
        <div style="font-size:42px; font-weight:800; color:#00E5FF; text-shadow:0 0 10px rgba(0,229,255,0.3);">
            {value}
        </div>
    </div>
    """

# ── MAIN VIEWPORT ──
if nav == "Tactical Map":
    l_col, r_col = st.columns([0.7, 0.3])

    with r_col:
        st.markdown("## 🚨 CrisisFlow Command Center")
        
        # Unified Metric Placeholders
        score_placeholder = st.empty()
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            survival_placeholder = st.empty()
            critical_placeholder = st.empty()
        with m_col2:
            deaths_placeholder = st.empty()
            response_placeholder = st.empty()

        # Initial Population (Prevent Black UI)
        if st.session_state.state:
            def get_info():
                env = st.session_state.get('env')
                return env._build_info() if env else {"score": 0.0, "survival_rate": 0, "dead": 0, "critical_rescued": 0, "avg_response_time": 0}
            
            info = get_info()
            score_placeholder.markdown(big_score_card(f"{info['score']:.4f}"), unsafe_allow_html=True)
            survival_placeholder.markdown(metric_card("Survival Rate", f"{info['survival_rate']}%", "#00ff88"), unsafe_allow_html=True)
            deaths_placeholder.markdown(metric_card("Fatalities", f"{info['dead']}", "#ff3b3b"), unsafe_allow_html=True)
            critical_placeholder.markdown(metric_card("Critical Saved", f"{info['critical_rescued']}", "#00e5ff"), unsafe_allow_html=True)
            response_placeholder.markdown(metric_card("Avg Response", f"{info['avg_response_time']}s", "#00e5ff"), unsafe_allow_html=True)
        else:
            score_placeholder.markdown(big_score_card("0.0000"), unsafe_allow_html=True)
            survival_placeholder.markdown(metric_card("Survival Rate", "0%", "#00ff88"), unsafe_allow_html=True)
            deaths_placeholder.markdown(metric_card("Fatalities", "0", "#ff3b3b"), unsafe_allow_html=True)
            critical_placeholder.markdown(metric_card("Critical Saved", "0", "#00e5ff"), unsafe_allow_html=True)
            response_placeholder.markdown(metric_card("Avg Response", "0s", "#00e5ff"), unsafe_allow_html=True)

        # Operational Intelligence
        st.markdown("### Operational Intelligence")
        log_placeholder = st.empty()

        # Analytics Bottom
        st.markdown("### Performance Stream")
        chart_placeholder = st.empty()

    with l_col:
        st.markdown("### 🗺️ Tactical Deployment Map")
        map_placeholder = st.empty()
        
        st.markdown("### Infrastructure Network")
        hosp_placeholder = st.empty()
        step_status = st.empty()

        # Initial Map Render
        if st.session_state.state and not st.session_state.running:
            state_dict = st.session_state.state.model_dump() if hasattr(st.session_state.state, "model_dump") else st.session_state.state
            prev_state_dict = st.session_state.prev_state.model_dump() if hasattr(st.session_state.prev_state, "model_dump") else st.session_state.get('prev_state')
            fig = draw_pydeck_map(state_dict, prev_state=prev_state_dict, alpha=1.0)
            map_placeholder.plotly_chart(fig, width="stretch", key="initial_load_tactical")

    # Simulation Inner Loop
    # Simulation Main Control Flow (Flicker-Free Continuous Update)
    if st.session_state.running and not st.session_state.paused:
        env = st.session_state.env
        agent = st.session_state.agent
        if 'log_history' not in st.session_state: st.session_state.log_history = []
        
        max_steps = {"easy": 500, "medium": 800, "hard": 1000}[diff.lower()]
        
        while st.session_state.running and not st.session_state.paused:
            state = st.session_state.state
            state_dict = state.model_dump() if hasattr(state, "model_dump") else state
            actions = agent.select_action(state)
            
            # Generate Decision Trace
            for action in actions:
                p_id = action['patient_id']
                a_id = action['ambulance_id']
                p_data = next((p for p in state_dict['patients'] if p['id'] == p_id), None)
                sev = p_data['severity'] if p_data else 1
                sev_label = ["Minor", "Serious", "Critical"][sev-1]
                log_time = time.strftime('%H:%M:%S')
                log_msg = f'<div class="log-entry">[{log_time}] <span class="log-ambulance">AMB-{a_id:02d}</span> → <span class="log-severity-{sev}">[{sev_label}]</span> Patient P{p_id:02d}</div>'
                st.session_state.log_history.insert(0, log_msg)
                if len(st.session_state.log_history) > 30: st.session_state.log_history.pop()

            next_state, reward, done, info = env.step(actions)
            
            # State Update (Memory)
            st.session_state.prev_state = state
            st.session_state.state = next_state
            
            reward_val = reward.score if hasattr(reward, 'score') else reward
            st.session_state.rewards.append(reward_val)
            
            # ──────── COMPONENT UPDATE (ZERO-REFRESH) ────────
            
            # 1. Mission Efficiency & Metrics (Unified Cards)
            score_placeholder.markdown(big_score_card(f"{info['score']:.4f}"), unsafe_allow_html=True)
            survival_placeholder.markdown(metric_card("Survival Rate", f"{info['survival_rate']}%", "#00ff88"), unsafe_allow_html=True)
            deaths_placeholder.markdown(metric_card("Fatalities", f"{info['dead']}", "#ff3b3b"), unsafe_allow_html=True)
            critical_placeholder.markdown(metric_card("Critical Saved", f"{info['critical_rescued']}", "#00e5ff"), unsafe_allow_html=True)
            response_placeholder.markdown(metric_card("Avg Response", f"{info['avg_response_time']}s", "#00e5ff"), unsafe_allow_html=True)
            
            # 2. Intelligence Feed (Scroll-Locked Box)
            log_content = "".join(st.session_state.log_history)
            log_placeholder.markdown(f'<div class="log-container">{log_content}</div>', unsafe_allow_html=True)
            
            # 3. Tactical Environment
            next_state_dict = next_state.model_dump() if hasattr(next_state, "model_dump") else next_state
            prev_state_dict = st.session_state.prev_state.model_dump() if hasattr(st.session_state.prev_state, "model_dump") else st.session_state.get('prev_state')
            fig = draw_pydeck_map(next_state_dict, prev_state=prev_state_dict, alpha=0.3)
            # Make the figure mathematically unique each step to bypass DuplicateElementId in the while loop
            fig.update_layout(title=f"<!-- step {next_state_dict['time_step']} -->")
            map_placeholder.plotly_chart(fig, width="stretch")
            
            # 4. Infrastructure Tracking
            with hosp_placeholder.container():
                for h in next_state_dict['hospitals']:
                    pct = h['current_load'] / h['capacity']
                    st.progress(pct, text=f"Station-{h['id']:02d}: {h['current_load']}/{h['capacity']}")
            
            # 5. Performance Trend
            if st.session_state.rewards:
                chart_placeholder.line_chart(st.session_state.rewards, color="#00d4ff")
                
            step_status.caption(f"SYSTEM STEP: {next_state_dict['time_step']} / {max_steps} | CLOCK: {time.strftime('%H:%M:%S')}")

            if done:
                st.session_state.running = False
                st.success(f"TACTICAL OPERATION COMPLETE: SCORE {info['score']:.4f}")
                break
            
            time.sleep(st.session_state.speed)


elif nav == "Agent Comparison":
    st.title("Autonomous Benchmarking Analytics")
    st.markdown("Evaluating the performance delta between the **Baseline AI Agent** and **Random Dispatch Selection**.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        c_diff = st.selectbox("Benchmark Complexity", ["Easy", "Medium", "Hard"], index=1, key="bench_diff")
    with col_b:
        c_seed = st.number_input("Benchmark Seed", min_value=0, value=42, key="bench_seed")
    
    if st.button("Initialize Benchmarking", type="primary"):
        with st.spinner("Executing Scenario Simulations..."):
            results = get_benchmarking_results(c_diff, c_seed)
            
            # Metrics Display
            m_col1, m_col2 = st.columns(2)
            for i, (name, inf) in enumerate(results.items()):
                with [m_col1, m_col2][i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader(f"Agent: {name.upper()}")
                    st.metric("Survival Rate", f"{inf['survival_rate']}%")
                    st.metric("Final Score", f"{inf['score']:.4f}")
                    st.metric("Fatalities", inf['dead'], delta_color="inverse")
                    st.markdown('</div>', unsafe_allow_html=True)

            # Chart
            comp_df = pd.DataFrame({
                "Metric": ["Survival Rate", "Final Score (x100)"],
                "Baseline": [results['Baseline']['survival_rate'], results['Baseline']['score'] * 100],
                "Random": [results['Random']['survival_rate'], results['Random']['score'] * 100]
            }).set_index("Metric")
            
            st.bar_chart(comp_df)
            
            diff_rate = results['Baseline']['survival_rate'] - results['Random']['survival_rate']
            if diff_rate > 0:
                st.success(f"**Analytics Result:** Baseline AI demonstrated a {diff_rate:.1f}% improvement in survival efficiency.")

elif nav == "Mission Logs":
    st.title("Operational Event Logs")
    st.info("System logs are automatically recorded during the tactical deployment phase. Start a simulation to view live event tracing.")
    if st.session_state.state:
        st.dataframe(pd.DataFrame(st.session_state.state['patients']), use_container_width=True)
    else:
        st.write("No active operation data found.")
