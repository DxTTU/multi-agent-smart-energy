# Project 2 — Multi-Agent Smart Energy Optimizer

## What it actually does
Simulates an energy-management workflow using five cooperating stages: Monitor -> Forecast -> Policy/RAG -> Optimizer -> Controller.

- Generates 24 hours of simulated energy readings.
- Uses a PyTorch LSTM component to produce a short-term forecast.
- Retrieves safety/energy rules from `policies.md` using RAG.
- Uses TensorFlow/Keras for risk scoring when TensorFlow is installed.
- Uses a deterministic NumPy fallback when TensorFlow is unavailable.
- Chooses MONITOR, RECOMMEND, or SWITCH_OFF_SIMULATED.
- The controller never touches real hardware; it only prints a simulated action.

## Run
```
pip install -r requirements.txt
python main.py
```

Smoke test: `python test_project.py`
