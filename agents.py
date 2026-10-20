from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float
    automatic: bool


class MonitorAgent:
    name = "MonitorAgent"

    def run(self):
        rng = np.random.default_rng(4)
        hours = np.arange(24)
        base = 1.0 + 0.45 * np.sin((hours - 7) / 24 * 2 * np.pi)
        return pd.DataFrame({
            "hour": hours,
            "kwh": np.maximum(base + rng.normal(0, .08, 24), .2),
            "appliance": ["washing_machine" if h in [21, 22, 23] else "general_load" for h in hours],
        })


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])


class ForecastAgent:
    name = "ForecastAgent"

    def __init__(self):
        torch.manual_seed(4)
        self.model = LSTMForecaster()

    def run(self, series):
        x = torch.tensor(series[-6:], dtype=torch.float32).view(1, 6, 1)
        with torch.no_grad():
            pred = self.model(x).item()
        return float(.5 * pred + .5 * series[-3:].mean())


class TensorFlowRiskModel:
    def __init__(self):
        X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype="float32")
        y = np.array([0, 0, 1, 1, 0, 1])
        if TF_AVAILABLE:
            self.backend = "TensorFlow/Keras"
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(3,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            self.model.compile(optimizer="adam", loss="binary_crossentropy")
            self.model.fit(X, y, epochs=25, verbose=0)
        else:
            self.backend = "NumPy fallback (TensorFlow unavailable in this runtime)"

    def risk(self, high_power, safety_critical, approved):
        if TF_AVAILABLE:
            return float(self.model.predict(
                np.array([[high_power, safety_critical, approved]], dtype="float32"),
                verbose=0)[0][0])
        # Deterministic fallback matching the policy intent: safety-critical/high-power raises risk; approval lowers it.
        return float(np.clip(.15 + .45 * high_power + .55 * safety_critical - .15 * approved, 0, 1))


class PolicyAgent:
    name = "PolicyAgent"

    def __init__(self, path):
        self.chunks = [x.strip() for x in open(path, encoding="utf-8").read().split("\n\n") if x.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.risk_model = TensorFlowRiskModel()

    def run(self, appliance, approved):
        q = self.vectorizer.transform([f"{appliance} automatic switching approval safety"])
        scores = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(scores)[::-1][:2]
        evidence = [self.chunks[i] for i in idx]
        safety = int(appliance in {"medical_equipment", "refrigerator"})
        high = int(appliance in {"water_heater", "air_conditioner"})
        return evidence, self.risk_model.risk(high, safety, int(approved))


class OptimizerAgent:
    name = "OptimizerAgent"

    def run(self, forecast, current, risk, approved):
        if risk > .55:
            return Decision("RECOMMEND", "Risk model indicates automatic control is not suitable.", 1 - risk, False)
        if approved and current > forecast * 1.10:
            return Decision("SWITCH_OFF_SIMULATED", "Forecast and policy evidence indicate a potentially wasteful load.", .80, True)
        return Decision("MONITOR", "No sufficiently strong low-risk saving action.", .70, False)


class ControllerAgent:
    name = "ControllerAgent"

    def run(self, d):
        return f"SIMULATED ACTION EXECUTED: {d.action} | {d.reason}" if d.automatic else f"NO HARDWARE ACTION: {d.action} | {d.reason}"


class SmartEnergySystem:
    def __init__(self, path="policies.md"):
        self.monitor = MonitorAgent()
        self.forecaster = ForecastAgent()
        self.policy = PolicyAgent(path)
        self.optimizer = OptimizerAgent()
        self.controller = ControllerAgent()

    def run(self):
        df = self.monitor.run()
        s = df.kwh.to_numpy()
        forecast = self.forecaster.run(s)
        current = float(s[-1])
        evidence, risk = self.policy.run("washing_machine", True)
        decision = self.optimizer.run(forecast, current, risk, True)
        return {
            "current_kwh": round(current, 3),
            "forecast_kwh": round(forecast, 3),
            "policy_risk": round(risk, 3),
            "decision": decision,
            "rag_evidence": evidence,
            "action": self.controller.run(decision),
            "tensorflow_backend": self.policy.risk_model.backend,
        }
