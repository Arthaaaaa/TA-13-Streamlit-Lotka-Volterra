import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ===============================
# JUDUL APLIKASI
# ===============================
st.title("Simulasi Predator–Prey (Lotka–Volterra)")
st.write("Dataset: Snowshoe Hare (Mangsa) & Lynx (Predator)")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("dataset.csv")
df.columns = [c.strip().lower() for c in df.columns]
df = df.sort_values("year").reset_index(drop=True)

SCALE = 1000.0
hare_data = df["hare"].values / SCALE
lynx_data = df["lynx"].values / SCALE

t = np.arange(len(df))

# ===============================
# SIDEBAR — SLIDER PARAMETER
# ===============================
st.sidebar.header("Parameter Model")

alpha = st.sidebar.slider(
    "α (pertumbuhan mangsa)", 0.0, 2.0, 0.6, 0.01
)
beta = st.sidebar.slider(
    "β (interaksi mangsa–predator)", 0.0, 0.1, 0.02, 0.001
)
gamma = st.sidebar.slider(
    "γ (kematian predator)", 0.0, 2.0, 0.5, 0.01
)
delta = st.sidebar.slider(
    "δ (pertumbuhan predator)", 0.0, 0.1, 0.01, 0.001
)

# ===============================
# MODEL LOTKA–VOLTERRA
# ===============================
def lotka_volterra(state, t, alpha, beta, gamma, delta):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# kondisi awal
initial_state = [hare_data[0], lynx_data[0]]

solution = odeint(
    lotka_volterra,
    initial_state,
    t,
    args=(alpha, beta, gamma, delta)
)

hare_sim = solution[:, 0]
lynx_sim = solution[:, 1]

# ===============================
# VISUALISASI
# ===============================
fig, ax = plt.subplots(figsize=(10, 5))

# data asli
ax.plot(t, hare_data, "b--", label="Hare (Data Asli)")
ax.plot(t, lynx_data, "r--", label="Lynx (Data Asli)")

# model
ax.plot(t, hare_sim, "b", label="Hare (Model)")
ax.plot(t, lynx_sim, "r", label="Lynx (Model)")

ax.set_xlabel("Waktu (tahun)")
ax.set_ylabel("Populasi (skala ribuan)")
ax.set_title("Model Lotka–Volterra vs Data Asli")
ax.legend()
ax.grid(True)

st.pyplot(fig)
