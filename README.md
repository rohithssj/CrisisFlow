# 🚑 CrisisFlow — AI-Powered Disaster Response Simulation

CrisisFlow is a **real-time crisis response simulation system** that models how emergency resources (ambulances, hospitals) are allocated under high-pressure scenarios using intelligent decision-making.

It is designed to demonstrate how **AI-driven coordination systems** can significantly improve survival outcomes during disasters.

---

# 🌍 Problem Statement

In real-world disasters (earthquakes, floods, urban accidents):

* Emergency resources are **limited**
* Decisions must be made **in seconds**
* Poor coordination leads to:

  * Increased fatalities
  * Delayed response times
  * Underutilized hospitals

👉 The challenge is:

> How can we intelligently allocate emergency resources in real-time to maximize survival?

---

# 💡 Our Approach

CrisisFlow simulates a **dynamic disaster environment** and uses an intelligent agent to:

* Evaluate multiple patients simultaneously
* Prioritize based on:

  * Severity
  * Distance
  * Waiting time
* Assign ambulances efficiently
* Route patients to optimal hospitals

---

# 🧠 System Architecture

```
Environment (Gym-style)
        ↓
State Representation
        ↓
Decision Agent (Rule-based → RL-ready)
        ↓
Multi-Dispatch System
        ↓
Simulation Loop
        ↓
Metrics + Visualization
```

---

# ⚙️ Core Features

## 🚨 1. Multi-Dispatch Decision System

* Assigns multiple ambulances per timestep
* Optimizes resource usage across the system

---

## 🎯 2. Intelligent Priority Scoring

Each patient is ranked using:

* Severity (critical > medium > stable)
* Distance from ambulance
* Waiting time

👉 Ensures **high-risk patients are prioritized first**

---

## 🌐 3. Dynamic Simulation Environment

* Randomized scenarios every run
* Traffic-like complexity
* Real-time state updates

---

## 🗺️ 4. Tactical Map Visualization (Advanced UI)

A **live command center dashboard** built with PyDeck:

* 🚑 Ambulances (icon-based, animated)
* 🏥 Hospitals (capacity-aware visualization)
* 🧑 Patients (color-coded by severity)
* ❌ Fatalities tracking
* 🔗 Route visualization (ambulance → patient → hospital)

---

## 🎬 5. Real-Time Animation Engine

* Smooth movement (no teleportation)
* Continuous updates without full page refresh
* Live system behavior visualization

---

## 🧠 6. Operational Intelligence Feed

A live decision log showing:

* Ambulance assignments
* Patient prioritization
* Real-time actions taken by the agent

👉 Makes AI decisions **transparent and interpretable**

---

## 📊 7. Performance Metrics Dashboard

### 🔥 Mission Efficiency Score

Overall system performance (0 → 1)

### 📦 Key Metrics

* Survival Rate
* Fatalities
* Critical Patients Saved
* Average Response Time

All metrics are displayed in **premium card-based UI**

---

## 📈 8. Reward Performance Tracking

* Tracks reward progression over time
* Helps evaluate agent performance

---

## ⚡ 9. High-Performance UI Architecture

* No full-page refreshes
* Component-level updates using placeholders
* Smooth real-time experience

---

# 🧪 Simulation Workflow

1. Initialize environment
2. Generate random crisis scenario
3. Agent observes system state
4. Assigns ambulances
5. Simulation updates:

   * Movement
   * Pickups
   * Deliveries
6. Metrics updated
7. Repeat until completion

---

# 🏗️ Tech Stack

* **Python**
* **Streamlit** (UI Dashboard)
* **PyDeck (Deck.gl)** — Map visualization
* **NumPy / Pandas** — Data handling

---

# 🔄 OpenEnv Compliance

CrisisFlow follows a structured environment design:

* `reset()` → Initialize scenario
* `step(action)` → Execute decisions
* `state` → Current system snapshot

---

# 📁 Project Structure

```
crisisflow/
│
├── agents/
│   └── baseline_agent.py
│
├── environment/
│   └── crisis_env.py
│
├── ui/
│   └── map.py
│
├── app.py
├── inference.py
├── Dockerfile
└── requirements.txt
```

---

# 🚀 Key Innovations

✅ Multi-dispatch coordination system
✅ Real-time tactical visualization
✅ Smooth animation engine (no refresh UX)
✅ Explainable AI decisions
✅ Production-grade dashboard UI

---

# 📦 Deployment Ready

* Docker-compatible
* Hugging Face Spaces ready
* Lightweight and efficient

---

# 🔮 Future Scope

* Reinforcement Learning-based agent
* Traffic-aware routing
* Real-world map integration
* Multi-city disaster scaling
* Predictive demand modeling

---

# 🏁 Conclusion

CrisisFlow demonstrates how **AI + simulation + real-time visualization** can transform disaster response systems.

> From reactive decision-making → to intelligent, coordinated action.

---

# 👨‍💻 Built For

* Meta / PyTorch Hackathon
* OpenEnv-style AI simulation challenges

---

# ⭐ Final Note

CrisisFlow is not just a simulation —
it is a **vision of how intelligent systems can save lives at scale**.
