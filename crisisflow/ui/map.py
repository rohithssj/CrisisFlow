import plotly.graph_objects as go
import numpy as np
import random


def draw_pydeck_map(state, prev_state=None, alpha=0.3):
    """
    Renders a premium Tactical Map using Plotly.
    Uses unicode icons for high-performance visualization of ambulances, patients, and hospitals.
    """
    # 0. Ensure dict format (handle Pydantic models)
    if hasattr(state, "model_dump"):
        state = state.model_dump()
    if prev_state is not None and hasattr(prev_state, "model_dump"):
        prev_state = prev_state.model_dump()

    fig = go.Figure()

    # 1. Coordinate Grid (Tactical Style)
    for i in np.linspace(0, 1, 11):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=1, line=dict(color="rgba(255,255,255,0.05)", width=1), layer="below")
        fig.add_shape(type="line", x0=0, y0=i, x1=1, y1=i, line=dict(color="rgba(255,255,255,0.05)", width=1), layer="below")

    # 2. Routes (Dashed Cyan Neon)
    for amb in state.get("ambulances", []):
        if amb.get("busy") and amb.get("target_patient_id") is not None:
            pat = next((p for p in state.get("patients", []) if p["id"] == amb["target_patient_id"]), None)
            if pat:
                fig.add_trace(go.Scatter(
                    x=[amb["x"], pat["x"]], y=[amb["y"], pat["y"]],
                    mode="lines", line=dict(color="#00E5FF", width=1.5, dash="dash"),
                    opacity=0.6, hoverinfo="skip"
                ))

    # 3. Hospitals (🏥 Icon + Label)
    for h in state.get("hospitals", []):
        fig.add_trace(go.Scatter(
            x=[h["x"]], y=[h["y"]], 
            mode="markers+text",
            marker=dict(size=40, color="rgba(0, 229, 255, 0.1)", line=dict(width=1, color="rgba(0, 229, 255, 0.3)")),
            text="🏥",
            textposition="middle center",
            textfont=dict(size=24),
            name="Infrastructure",
            hovertext=f"<b>Hospital ST-{h['id']:02d}</b><br>Capacity: {h['current_load']}/{h['capacity']}",
            hoverinfo="text"
        ))

    # 4. Patients (🧑 icons - Severity coded)
    severity_map = {
        3: {"icon": "🧑", "color": "#FF3B3B", "label": "CRITICAL"},
        2: {"icon": "🧑", "color": "#FFA500", "label": "SERIOUS"},
        1: {"icon": "🧑", "color": "#00FF88", "label": "STABLE"}
    }
    
    for p in state.get("patients", []):
        if p.get("rescued"): continue
        
        icon = "❌" if p.get("dead") else "🧑"
        color = "#9CA3AF" if p.get("dead") else severity_map.get(p["severity"], severity_map[1])["color"]
        
        fig.add_trace(go.Scatter(
            x=[p["x"]], y=[p["y"]],
            mode="markers+text",
            marker=dict(size=25 if p.get("dead") else 30, color="rgba(0,0,0,0)", line=dict(width=0)),
            text=icon,
            textposition="middle center",
            textfont=dict(size=20, color=color),
            hovertext=f"<b>Patient {p['id']}</b><br>Status: {'DEAD' if p.get('dead') else severity_map.get(p['severity'])['label']}<br>Wait Time: {p['time_waiting']} steps",
            hoverinfo="text"
        ))

    # 5. Ambulances (🚑 Icon + Cyan Glow)
    for amb in state.get("ambulances", []):
        cur_x, cur_y = amb["x"], amb["y"]
        if prev_state:
            prev_amb = next((a for a in prev_state.get("ambulances", []) if a["id"] == amb["id"]), None)
            if prev_amb:
                # Smooth interpolation
                cur_x = prev_amb["x"] + alpha * (amb["x"] - prev_amb["x"])
                cur_y = prev_amb["y"] + alpha * (amb["y"] - prev_amb["y"])

        fig.add_trace(go.Scatter(
            x=[cur_x], y=[cur_y], 
            mode="markers+text",
            marker=dict(size=35, color="rgba(0, 229, 255, 0.15)", line=dict(width=1, color="#00E5FF")),
            text="🚑",
            textposition="middle center",
            textfont=dict(size=22),
            hovertext=f"<b>Ambulance A{amb['id']:02d}</b><br>Status: {'BUSY' if amb.get('busy') else 'IDLE'}<br>Zone: {amb.get('traffic_zone', 'N/A')}",
            hoverinfo="text"
        ))

    # 6. Final Layout Polish
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0B0F14",
        paper_bgcolor="#0B0F14",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        height=680,
        dragmode=False,
        hoverlabel=dict(bgcolor="#1F2937", font_size=12, font_family="Inter")
    )
    
    return fig
