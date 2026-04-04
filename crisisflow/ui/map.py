import plotly.graph_objects as go
import numpy as np
import random


def draw_pydeck_map(state, prev_state=None, alpha=0.2):
    """
    Renders a high-fidelity Tactical Map using Plotly for maximum stability.
    Uses images for icon support and neon styling.
    """
    fig = go.Figure()

    # 1. Coordinate Grid
    for i in np.linspace(0, 1, 11):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=1, line=dict(color="rgba(255,255,255,0.08)", width=1), layer="below")
        fig.add_shape(type="line", x0=0, y0=i, x1=1, y1=i, line=dict(color="rgba(255,255,255,0.08)", width=1), layer="below")

    # 2. Routes (Dashed Neon)
    for amb in state.get("ambulances", []):
        if amb.get("busy") and amb.get("target_patient_id") is not None:
            pat = next((p for p in state.get("patients", []) if p["id"] == amb["target_patient_id"]), None)
            if pat:
                fig.add_trace(go.Scatter(
                    x=[amb["x"], pat["x"]], y=[amb["y"], pat["y"]],
                    mode="lines", line=dict(color="#00d4ff", width=1, dash="dash"),
                    opacity=0.4, hoverinfo="skip"
                ))

    # 3. Hospitals (Robust Text Markers + Glow)
    for h in state.get("hospitals", []):
        # Stable, thread-safe local RNG for offset
        local_rng = random.Random(h["id"])
        offset_x = h["x"] + local_rng.uniform(-0.02, 0.02)
        offset_y = h["y"] + local_rng.uniform(-0.02, 0.02)
        
        # Combined Trace: Glow Circle + Hospital Icon Text
        fig.add_trace(go.Scatter(
            x=[offset_x], y=[offset_y], 
            mode="markers+text",
            marker=dict(size=45, color="rgba(0, 100, 255, 0.15)", line=dict(width=1, color="rgba(0, 150, 255, 0.3)")),
            text="🏥",
            textposition="middle center",
            textfont=dict(size=22),
            name="Infrastructure",
            hovertext=f"Station-{h['id']:02d}<br>Load: {h['current_load']}/{h['capacity']}",
            hoverinfo="text"
        ))

    # 4. Patients (Severity Coded Pins)
    severity_cfg = {
        3: {"color": "#ff3e3e", "name": "Critical", "size": 15},
        2: {"color": "#ffa500", "name": "Serious", "size": 12},
        1: {"color": "#00ff88", "name": "Stable", "size": 10}
    }
    for p in state.get("patients", []):
        if p.get("rescued") or p.get("dead"): continue
        cfg = severity_cfg.get(p["severity"], severity_cfg[1])
        fig.add_trace(go.Scatter(
            x=[p["x"]], y=[p["y"]], mode="markers",
            marker=dict(size=cfg["size"], color=cfg["color"], line=dict(width=1, color="white")),
            name=cfg["name"],
            text=f"Patient {p['id']}<br>Wait: {p['time_waiting']}s",
            hoverinfo="text"
        ))

    # 5. Ambulances (Text Markers)
    for amb in state.get("ambulances", []):
        cur_x, cur_y = amb["x"], amb["y"]
        if prev_state:
            prev_amb = next((a for a in prev_state.get("ambulances", []) if a["id"] == amb["id"]), None)
            if prev_amb:
                cur_x = prev_amb["x"] + alpha * (amb["x"] - prev_amb["x"])
                cur_y = prev_amb["y"] + alpha * (amb["y"] - prev_amb["y"])

        fig.add_trace(go.Scatter(
            x=[cur_x], y=[cur_y], 
            mode="text",
            text="🚑",
            textposition="middle center",
            textfont=dict(size=22),
            name="Unit",
            hovertext=f"Ambulance {amb['id']}<br>Status: {'DEPLOYED' if amb.get('busy') else 'READY'}",
            hoverinfo="text"
        ))

    # Layout Aesthetics
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#080a0f",
        paper_bgcolor="#080a0f",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        height=720,
        dragmode=False
    )
    
    return fig
