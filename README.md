# ⚡ CrisisFlow | Emergency Response Command Center

CrisisFlow is a high-performance emergency response simulator designed for real-time tactical decision-making and agent benchmarking.

---

## 🛠️ Requirements

- **Python Version:** Python **3.11** only (required for stability and DLL compatibility on Windows)
- **OS:** Windows / Linux / macOS
- **Containerization:** Docker (optional)

---

## 🚀 Local Setup

Follow these steps to get CrisisFlow running on your local machine:

### 1. Install Python 3.11
Ensure you have Python 3.11 installed. You can download it from [python.org](https://www.python.org/downloads/windows/).

### 2. Create Virtual Environment
Open your terminal and run:
```bash
# Using the Python Launcher
py -3.11 -m venv venv
```

### 3. Activate & Install Requirements
```bash
# Windows
.\venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 🐳 Docker Usage

CrisisFlow is fully containerized and production-ready.

### Build Image
```bash
docker build -t crisisflow .
```

### Run Container
```bash
docker run -p 8501:8501 crisisflow
```

---

## 🔒 Troubleshooting (Windows DLL Errors)

If you encounter an `ImportError: DLL load failed` when importing `pandas`:

1. **Verify Python Version:** Ensure you are NOT using Python 3.13. Switch to **Python 3.11**.
2. **Clean Reinstall:**
   ```bash
   pip uninstall pandas -y
   pip install pandas --no-cache-dir
   ```
3. **Environment Isolation:** Ensure you are using a virtual environment (`venv`) to avoid conflicts with global packages.

---

## 🌍 Compatibility

CrisisFlow is designed to be highly portable and is guaranteed compatible with:
- ✅ **Docker**
- ✅ **Hugging Face Spaces**
- ✅ **Streamlit Cloud**

---

## 📄 License
This project is part of the Meta-Hackathon series. 🚀
