# QUAD_CORE_AI 🧠

**Financial Market Forecasting System Based on Cyclic Consensus**

QUAD_CORE_AI is an advanced algorithmic system designed to analyze the BTC/USDT market. The project breaks away from the single-model approach, creating a "Council of Algorithms." The system's uniqueness lies in the synthesis of four fundamentally different mathematical approaches and the application of a deterministic, **cyclic consensus inversion strategy**.

---

## 🏗 The Four Pillars of Architecture (The Quad Core)

The project utilizes four fundamentally different data analysis approaches. This ensures that errors from one model are corrected by the others.

### 1. Monte Carlo (Stochastic Simulation) 🎲
*   **Characteristics:** A stochastic simulation method. The model does not "predict" a single price but generates thousands of possible future paths based on volatility and probability distributions.
*   **Role in the System:** Determines **pure statistical probability**.
    *   Visible in logs as parameters: `Win` (how many paths ended in profit), `Ahead` (time lookahead), `Sup` (support level).
    *   If Monte Carlo shows a 70% chance of growth, it provides a strong statistical foundation for the other models.

### 2. Random Forest (Ensemble Learning) 🌳
*   **Characteristics:** An ensemble of decision trees that creates non-linear rules based on raw historical data.
*   **Role in the System:** The Stabilizer ("Anchor").
    *   Distinguished by the **Trust Score** mechanism. The raw result (`Raw`) is corrected by historical accuracy (`Acc`).
    *   If the model has been wrong in the past, its weight is drastically reduced (e.g., `x -0.03 [Trust]`), preventing the system from following false signals.

### 3. KAN v2.0 (Kolmogorov-Arnold Networks) 🧬
*   **Characteristics:** A modern alternative to classical MLP networks. Instead of fixed activation functions at nodes, KAN possesses learnable activation functions (splines) on the edges.
*   **Role in the System:** The Mathematical Genius.
    *   Excellent approximation of complex, non-linear functions. It captures subtle mathematical dependencies in price action that are invisible to standard neural networks and statistical methods.

### 4. Neural Trend (Direct Network Output) 📉
*   **Characteristics:** A Deep Learning neural network optimized for detecting momentum and trend direction.
*   **Role in the System:** Momentum Detection.
    *   Acts as a "compass" indicating whether the market is currently in a strong growth or decline phase, without delving into volatility details.

---

## ⚙️ Decision Strategy: Cyclic Inversion (The Cycle)

The system does not blindly trust the final result. Instead, it applies a deterministic strategy of changing the trading vector over time to exploit the nature of markets (which sometimes follow the trend and sometimes break it).

### Step 1: Calculating Weighted Consensus (Probability of BUY)

All models return a percentage value representing the **Probability of Growth (Long Probability)**.
*   Value **> 50%** indicates an upward trend.
*   Value **< 50%** indicates a downward trend.

**Example of Consensus Calculation:**
1.  **Monte Carlo:** 70.0% (Strong BUY signal)
2.  **Random Forest:** 50.0% (Neutral)
3.  **KAN v2.0:** 44.5% (Slight SELL signal)
4.  **Neural Trend:** 5.1% (Very strong SELL signal)

**Consensus Result:** The weighted average is, for example, **42.4%**.
Since 42.4% < 50%, the system interprets this as a **SELL** signal.

### Step 2: The Game Cycle (Follow vs. Invert)
After establishing the Consensus, the system checks which phase of the cycle it is currently in. This protects against market traps.

*   **Phase 1: Aligned (Normal Mode):**
    *   The system plays **in accordance** with the Consensus.
    *   Consensus says SELL -> System opens **SHORT**.
*   **Phase 2: Inversion (Inverted Mode):**
    *   The system plays **against** the Consensus (multiplier `x-1`).
    *   Consensus says SELL -> System opens **LONG**.
    *   *Logic:* We assume that in this phase the market will "trick" most indicators (Fakeout), so the "wrong" signal from the algorithms becomes the ideal reverse signal.

---

## 🚀 Startup and Architecture

### ⚙️ Production Configuration (Systemd)

The system runs as a background service (`service`), ensuring auto-start after reboot and automatic process recovery in case of failure. The configuration has been optimized to share the AI model state (single process) while handling multiple users simultaneously (multithreading).

**File location:** `/etc/systemd/system/quad_core.service`

```ini
[Unit]
Description=Gunicorn instance to serve QUAD_CORE_AI
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/root/QUAD_CORE_AI
Environment="PATH=/root/QUAD_CORE_AI/venv/bin"
# Force immediate log writing (without RAM buffering)
Environment="PYTHONUNBUFFERED=1"

# ============================================================
# KEY PERFORMANCE PARAMETERS:
# -w 1         -> ONE Worker process.
#                 Guarantees that all users see the same results (shared memory state).
#                 Saves RAM (AI models are loaded only once).
# --threads 16 -> SIXTEEN threads.
#                 Allows handling multiple HTTP requests simultaneously within a single process.
# --timeout 120 -> Increased timeout (for long AI calculations).
# ============================================================

ExecStart=/root/QUAD_CORE_AI/venv/bin/gunicorn -w 1 --threads 16 --timeout 120 -b 127.0.0.1:8050 app:app

# Redirect all logs (print and error) to a file in the project folder
StandardOutput=append:/root/QUAD_CORE_AI/app.log
StandardError=append:/root/QUAD_CORE_AI/app.log

# Automatic restart after failure with a 5-second delay (protection against restart loops)
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### The Role of `app.py`

The heart of the system is the **`app.py`** file. It acts as the orchestrator connecting the world of AI calculations with the web world.

1.  **Thread Orchestrator:** On startup, it launches the `background_worker` – an independent background thread that:
    *   Fetches market data.
    *   Runs predictive processes (MC, RF, KAN, Trend).
    *   Manages the cycle and inversion logic.
2.  **API Server:** Exposes calculation results for the browser interface.

---

## 🛠 Startup Procedure

The system is designed for unattended operation.

**1. Start the system:**
```bash
systemctl start quad_core
```

**2. Verify operation (View the AI Brain):**
To see live how models are voting and what cycle phase we are in:
```bash
tail -f /root/QUAD_CORE_AI/app.log
```

**3. Restart (after code changes):**
```bash
systemctl restart quad_core
```

**4. Stop:**
```bash
systemctl stop quad_core
```

### Developer Mode (Manual)
To run the system manually (e.g., for testing), bypassing the background service:
```bash
systemctl stop quad_core  # Free up the port
cd /root/QUAD_CORE_AI
source venv/bin/activate
python3 app.py
```

---

## 📊 Web Access

The dashboard visualizing model decisions and the current cycle state (Inverted/Normal) is available at the server IP address (port 80 forwarded by Nginx to 8050):

`http://91.107.236.122`

---
