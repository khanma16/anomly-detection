# Real-Time Network Anomaly Detection System

**Masters Research Project - Live ML-Based Cybersecurity System**

Real-time network security using machine learning with **LIVE packet capture**, trained models, and comprehensive anomaly detection.

## **Quick Start**

### **1. Setup Environment**
```bash
# Clone and setup
git clone <repository>
cd anomly-detection
python -m venv venv

venv\Scripts\activate
pip install -r requirements.txt
```

### **2. Train Models (One-time)**
```bash
python src/train.py --dataset all
```

### **3. Run Live System**
```bash
# Linux/Mac
sudo venv/bin/python app.py

# Windows (Run as Administrator)
venv\Scripts\python.exe app.py

# Open: http://localhost:8000
```

### **4. Generate Test Attacks**
```bash
# Linux/Mac (separate terminal)
sudo venv/bin/python attack_simulator.py --safe

# Windows (separate terminal)
venv\Scripts\python.exe attack_simulator.py --safe
```

### **System Requirements**
- **Linux/Mac**: sudo privileges for packet capture
- **Windows**: Administrator privileges
- **Python**: 3.8+
- **Dependencies**: PyShark, Scapy, scikit-learn, Flask

---

## **Project Structure**
```
anomly-detection/
├── app.py                     # Main web application
├── attack_simulator.py        # Attack generation tool
├── src/
│   ├── train.py              # Model training
│   ├── model_loader.py       # Model management
│   ├── alert_system.py       # Email notifications
│   └── performance_evaluation.py  # Model evaluation
├── models/                    # Trained ML models
├── config.yaml               # System configuration
└── requirements.txt          # Dependencies
```

---