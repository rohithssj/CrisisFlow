# 🚑 CrisisFlow

**CrisisFlow** is an advanced, modular simulation environment designed for developing and evaluating autonomous reinforcement learning (RL) agents for urban emergency response.

### 🌟 Project Overview
In a city grid, crises occur at random locations with varying severity levels. **CrisisFlow** simulates the complex task of dispatching emergency resources (ambulances) to patients while optimizing for survival rates, response times, and hospital load balancing.

### 🧩 Problem Statement
Traditional emergency response systems often struggle with peak demand and hospital bottlenecks. **CrisisFlow** provides a sandbox to train AI agents that can:
- **Prioritize** critical patients over minor incidents.
- **Strategically Dispatch** ambulances based on Euclidean travel time.
- **Balance Loads** between multiple hospitals to prevent system saturation.

---

### 🕹️ How It Works

#### **1. Environment Mechanics**
- **Patients**: Spawn with 3 severity levels (1=Minor, 3=Critical). Wait time increases until a death threshold is reached if not rescued.
- **Ambulances**: Fixed locations initially. Become "busy" during travel and rescue phases.
- **Hospitals**: Fixed locations with finite capacity. Patients are admitted and discharged over time.

#### **2. API & Format**
- **Action Space**: `{"ambulance_id": int, "patient_id": int}` - Dispatch a specific ambulance to a specific patient.
- **State Space**: A detailed dictionary nested with list of patients, ambulances, and global stats.
- **Reward Function**: Proximal rewards for successful rescues, scaled by severity and speed, with penalties for idle resources and patient fatalities.

---

### 🚀 Getting Started

#### **Local Execution**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/CrisisFlow.git
   cd CrisisFlow
   ```
2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Simulation**:
   ```bash
   python inference.py
   ```

#### **Run with Docker**
Simply build and run the provided container:
```bash
docker build -t crisisflow .
docker run crisisflow
```

---

### 📊 Example Output
```text
Step   | Action                    | Reward   | Active   | Rescued  | Dead    
--------------------------------------------------------------------------------
1      | Amb 0 -> Pat 0            | 0.0500   | 11       | 0        | 0       
2      | Amb 1 -> Pat 1            | 0.0500   | 11       | 0        | 0       
...
150    | Amb 2 -> Pat 34           | 0.1242   | 0        | 32       | 3       

--- SIMULATION COMPLETE ---
Final Score:       0.8422
Total Rescued:     32
Total Dead:        3
Total Steps:       150
```

---

### 🛠️ Tech Stack
- **Core Engine**: Python 3.10
- **Math & Logic**: NumPy, Gymnasium
- **Config Management**: PyYAML
- **Deployment**: Docker (Linux-slim)

---

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
