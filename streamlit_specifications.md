
# Streamlit Application Specification: RL for Optimal Trade Execution

## 1. Application Overview

The **RL for Optimal Trade Execution** Streamlit application is designed to demonstrate how a Reinforcement Learning (RL) agent can optimize the execution of large block orders in financial markets, minimizing costs compared to traditional algorithms. This application targets CFA Charterholders and Investment Professionals, specifically fulfilling the needs of a persona like Sarah Chen, Head of Trading at Alpha Capital Partners.

The application guides Sarah through a simulated real-world workflow:
1.  **Environment Setup**: Sarah configures a simulated market environment, inspired by the Almgren-Chriss framework, to model market impact.
2.  **Baseline Benchmarking**: She evaluates the performance of traditional execution algorithms, Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP), to establish a baseline.
3.  **DQN Agent Training**: Sarah initiates the training of a Deep Q-Network (DQN) agent within the simulated environment.
4.  **Performance Evaluation**: She rigorously compares the RL agent's performance against TWAP and VWAP using statistical metrics and visualizes its adaptive execution behavior.
5.  **Governance Assessment**: Finally, the application outlines critical governance and risk management considerations for deploying such a Tier 1 autonomous trading model in a production environment.

The core objective is to showcase the tangible ROI of AI in financial trading by demonstrating how an adaptive RL agent can achieve significant cost savings and more consistent execution compared to rigid, rule-based strategies.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source import (
    ExecutionEnvironment,
    calculate_implementation_shortfall,
    execute_twap,
    execute_vwap,
    DQNExecutionAgent,
    evaluate_agent,
    analyze_adaptive_behavior,
    ttest_ind
)
```

### `st.session_state` Design

`st.session_state` is used extensively to maintain the application's state across user interactions and page navigations, preventing re-computation of heavy tasks and ensuring continuity.

**Initialization**: All `st.session_state` keys are initialized at the beginning of the `app.py` script.

```python
if 'page' not in st.session_state:
    st.session_state['page'] = 'Introduction' # Controls sidebar navigation
if 'env_params' not in st.session_state:
    st.session_state['env_params'] = {
        'total_shares': 500000,
        'n_periods': 20,
        'initial_price': 150.0,
        'volatility': 0.015,
        'permanent_impact_coeff': 0.00005,
        'temporary_impact_coeff': 0.0005,
        'price_drift_mean': 0.0, # Added for completeness as per source.py init
        'price_drift_std': 0.0001  # Added for completeness as per source.py init
    }
if 'n_trials' not in st.session_state:
    st.session_state['n_trials'] = 100
if 'base_env_instance' not in st.session_state:
    st.session_state['base_env_instance'] = None
if 'dqn_agent' not in st.session_state:
    st.session_state['dqn_agent'] = None
if 'twap_costs' not in st.session_state:
    st.session_state['twap_costs'] = []
if 'vwap_costs' not in st.session_state:
    st.session_state['vwap_costs'] = []
if 'rl_costs' not in st.session_state:
    st.session_state['rl_costs'] = []
if 'twap_schedules' not in st.session_state:
    st.session_state['twap_schedules'] = []
if 'vwap_schedules' not in st.session_state:
    st.session_state['vwap_schedules'] = []
if 'rl_schedules' not in st.session_state:
    st.session_state['rl_schedules'] = []
if 'dqn_rewards_history' not in st.session_state:
    st.session_state['dqn_rewards_history'] = []
if 'rl_behavior_df' not in st.session_state:
    st.session_state['rl_behavior_df'] = pd.DataFrame()
if 'has_env_configured' not in st.session_state:
    st.session_state['has_env_configured'] = False
if 'has_baselines_run' not in st.session_state:
    st.session_state['has_baselines_run'] = False
if 'has_agent_trained' not in st.session_state:
    st.session_state['has_agent_trained'] = False
if 'has_agent_evaluated' not in st.session_state:
    st.session_state['has_agent_evaluated'] = False
```

**Update and Read Mechanisms**:

*   **`env_params`**: Updated by user input widgets (sliders/number inputs) on the 'Environment Setup' page. Read when `ExecutionEnvironment` is instantiated and when `has_env_configured` is checked.
*   **`n_trials`**: Updated by a number input on the 'Baseline Algorithms' page. Read when baseline functions (`execute_twap`, `execute_vwap`) and `evaluate_agent` are called.
*   **`base_env_instance`**: Initialized with `ExecutionEnvironment(**st.session_state['env_params'])` on 'Environment Setup' page. Passed to `execute_twap`, `execute_vwap`, `DQNExecutionAgent`, `evaluate_agent`, `analyze_adaptive_behavior`.
*   **`dqn_agent`**: Initialized with `DQNExecutionAgent(...)` and updated after `dqn_agent.train()` on 'DQN Agent Training' page. Read by `evaluate_agent` and `analyze_adaptive_behavior`.
*   **`twap_costs`, `vwap_costs`, `rl_costs`**: Updated with lists of costs after respective execution functions are called. Read for displaying performance tables and plots.
*   **`twap_schedules`, `vwap_schedules`, `rl_schedules`**: Updated with lists of execution histories after respective functions. Read for plotting execution schedules.
*   **`dqn_rewards_history`**: Updated after `dqn_agent.train()` completes. Read for plotting the learning curve.
*   **`rl_behavior_df`**: Updated after `analyze_adaptive_behavior` is called. Read for plotting adaptive behavior.
*   **`has_env_configured`, `has_baselines_run`, `has_agent_trained`, `has_agent_evaluated`**: Boolean flags, set to `True` upon successful completion of each major step. Used to enable subsequent steps and control conditional rendering of warnings/sections.

### Application Structure and Flow

The application simulates a multi-page experience using a Streamlit sidebar `st.selectbox` for navigation.

```python
st.sidebar.title("Navigation")
st.session_state['page'] = st.sidebar.selectbox(
    "Go to",
    [
        "Introduction",
        "1. Environment Setup",
        "2. Baseline Algorithms",
        "3. DQN Agent Training",
        "4. Performance Evaluation",
        "5. Governance & Deployability"
    ]
)

# Conditional rendering of pages based on st.session_state['page']
```

---

#### Page: Introduction

**Markdown**:
```python
st.title("RL for Optimal Trade Execution")
st.markdown(f"## Case Study: Optimizing Large Block Order Execution at an Institutional Asset Manager")
st.markdown(f"**Persona:** Sarah Chen, Head of Trading at \"Alpha Capital Partners\"")
st.markdown(f"")
st.markdown(f"**Scenario:** Alpha Capital Partners frequently executes large block orders, often exceeding 500,000 shares for a single stock. Sarah's mandate is to minimize execution costs and market impact. Traditional algorithms like TWAP and VWAP are rigid and often lead to significant market impact and implementation shortfall, especially in volatile or illiquid markets. Sarah is exploring cutting-edge AI solutions, specifically Reinforcement Learning (RL), to develop a more adaptive trading agent. She needs to understand its performance, analyze its behavior, and critically assess its deployability and governance requirements for a Tier 1 autonomous trading model.")
st.markdown(f"This application will guide Sarah through the process of building, training, evaluating, and analyzing an RL-based trading agent in a simulated market environment, demonstrating how AI can drive tangible ROI in financial trading.")
st.markdown(f"---")
st.subheader("Key Concepts")
st.markdown(f"**RL Agent's Value:** The RL agent's value is in *adaptation*, not *prediction*. It learns when to trade more and when to trade less based on observable market conditions—remaining inventory, time pressure, recent price drift, and inferred liquidity. This is a control problem (\"how to act optimally given current state\") rather than a prediction problem (\"what will the price be tomorrow\").")
st.markdown(f"**Expected Improvement:** RL agents can offer 3-10 basis points (bps) improvement per trade over TWAP. On a $1B daily trading desk, that translates to $300K-$1M per day in execution cost savings – one of the most concrete, measurable ROI calculations for AI in finance.")
```

---

#### Page: 1. Environment Setup

**Markdown**:
```python
st.title("1. Market Execution Environment Setup")
st.markdown(f"Sarah understands that a key challenge in executing large orders is **market impact**. To accurately test new strategies, we need a simulated environment that captures these dynamics. We'll use an `ExecutionEnvironment` class, inspired by the **Almgren-Chriss framework**, modeling both temporary and permanent market impact.")
st.markdown(f"")
st.subheader("Almgren-Chriss Market Impact Model")
st.markdown(f"The model incorporates:")

st.markdown(r"$$ \Delta P_{{\text{{temp}},t}} = \eta \cdot n_t $$")
st.markdown(r"where $\Delta P_{{\text{{temp}},t}}$ is the temporary price deviation, $\eta$ is the temporary impact parameter, and $n_t$ is the number of shares traded at time $t$.")

st.markdown(r"$$ \Delta P_{{\text{{perm}},t}} = \gamma \cdot n_t $$")
st.markdown(r"where $\Delta P_{{\text{{perm}},t}}$ is the permanent shift in the underlying mid-price, $\gamma$ is the permanent impact parameter, and $n_t$ is the number of shares traded at time $t$.")

st.markdown(f"")
st.markdown(f"The **implementation shortfall (IS)**, which the agent aims to minimize, is defined as:")

st.markdown(r"$$ IS = \frac{{\sum_{{t=1}}^{{T}} n_t \cdot (P_{{\text{{exec}},t}} - P_0)}}{{Q \cdot P_0}} \times 10000 \text{{ (in basis points)}} $$")
st.markdown(r"where $n_t$ are shares traded at time $t$, $P_{{\text{{exec}},t}}$ is the execution price, $P_0$ is the initial decision price, $Q$ is the total shares, and $T$ is total periods.")

st.markdown(f"Our reward will be $-IS$, so maximizing the reward means minimizing the execution cost.")
st.markdown(f"")
st.subheader("Configure Environment Parameters")
```

**Widgets**:
```python
# Use st.session_state['env_params'] for initial values and updates
st.session_state['env_params']['total_shares'] = st.number_input(
    "Total Shares to Execute (Q)", min_value=10000, max_value=5000000, value=st.session_state['env_params']['total_shares'], step=10000
)
st.session_state['env_params']['n_periods'] = st.slider(
    "Number of Execution Periods (T)", min_value=10, max_value=100, value=st.session_state['env_params']['n_periods']
)
st.session_state['env_params']['initial_price'] = st.number_input(
    "Initial Price ($P_0$)", min_value=50.0, max_value=500.0, value=st.session_state['env_params']['initial_price'], format="%.2f"
)
st.session_state['env_params']['volatility'] = st.slider(
    "Volatility (σ)", min_value=0.001, max_value=0.05, value=st.session_state['env_params']['volatility'], format="%.4f"
)
st.session_state['env_params']['permanent_impact_coeff'] = st.slider(
    "Permanent Impact Coeff (γ)", min_value=0.00001, max_value=0.0001, value=st.session_state['env_params']['permanent_impact_coeff'], format="%.5f"
)
st.session_state['env_params']['temporary_impact_coeff'] = st.slider(
    "Temporary Impact Coeff (η)", min_value=0.0001, max_value=0.001, value=st.session_state['env_params']['temporary_impact_coeff'], format="%.5f"
)
st.session_state['env_params']['price_drift_mean'] = st.slider(
    "Price Drift Mean", min_value=-0.0001, max_value=0.0001, value=st.session_state['env_params']['price_drift_mean'], format="%.5f"
)
st.session_state['env_params']['price_drift_std'] = st.slider(
    "Price Drift Std", min_value=0.00001, max_value=0.0002, value=st.session_state['env_params']['price_drift_std'], format="%.5f"
)
```

**Function Invocation / Interaction**:
```python
if st.button("Configure Environment"):
    with st.spinner("Initializing environment..."):
        try:
            st.session_state['base_env_instance'] = ExecutionEnvironment(**st.session_state['env_params'])
            obs, info = st.session_state['base_env_instance'].reset()
            st.session_state['has_env_configured'] = True
            st.success("Environment configured successfully!")
            st.markdown(f"### Current Environment Parameters:")
            st.markdown(f"Initial price: ${st.session_state['base_env_instance'].P0:.2f}, Volatility: {st.session_state['base_env_instance'].sigma:.1%}")
            st.markdown(f"Total shares: {st.session_state['base_env_instance'].Q:,}, Periods: {st.session_state['base_env_instance'].T}")
            st.markdown(f"Permanent Impact (γ): {st.session_state['base_env_instance'].gamma:.5f}, Temporary Impact (η): {st.session_state['base_env_instance'].eta:.5f}")
            st.markdown(f"Initial Observation (normalized remaining shares, time, price drift, volatility): {obs}")
            st.markdown(f"Action space mapping (fraction of remaining shares): {st.session_state['base_env_instance'].action_map}")
        except Exception as e:
            st.error(f"Error configuring environment: {e}")
            st.session_state['has_env_configured'] = False
```

---

#### Page: 2. Baseline Algorithms

**Markdown**:
```python
st.title("2. Benchmarking with Traditional Execution Algorithms")
st.markdown(f"Before trusting an adaptive AI agent, Sarah requires a clear benchmark. Our firm's standard practice relies on traditional algorithms like Time-Weighted Average Price (**TWAP**) and Volume-Weighted Average Price (**VWAP**). These algorithms are simple but lack adaptability.")
st.markdown(f"")
st.markdown(f"**TWAP** aims to execute an order by splitting it equally over a specified time period. For $Q$ total shares over $T$ periods, shares per period are $n_t = Q / T$.")
st.markdown(f"**VWAP** attempts to execute an order proportional to the expected market volume distribution. For $Q$ total shares, if the expected volume fraction at time $t$ is $v_t$, then $n_t = Q \cdot v_t$. We use a conceptual U-shaped volume profile here.")
st.markdown(f"We will run each strategy over multiple simulated trials to capture market variability and derive statistically robust performance metrics.")
st.markdown(f"")
```

**Widgets**:
```python
if not st.session_state['has_env_configured']:
    st.warning("Please configure the environment parameters first on the 'Environment Setup' page.")
else:
    st.session_state['n_trials'] = st.number_input(
        "Number of Simulation Trials", min_value=10, max_value=500, value=st.session_state['n_trials'], step=10
    )
```

**Function Invocation / Interaction**:
```python
    if st.button("Run Baselines"):
        with st.spinner(f"Running {st.session_state['n_trials']} trials for TWAP and VWAP... This might take a moment."):
            try:
                twap_costs = []
                vwap_costs = []
                twap_schedules_temp = []
                vwap_schedules_temp = []

                for i in range(st.session_state['n_trials']):
                    cost_twap, schedule_twap = execute_twap(st.session_state['base_env_instance'])
                    twap_costs.append(cost_twap)
                    twap_schedules_temp.append(schedule_twap)

                    cost_vwap, schedule_vwap = execute_vwap(st.session_state['base_env_instance'])
                    vwap_costs.append(cost_vwap)
                    vwap_schedules_temp.append(schedule_vwap)

                st.session_state['twap_costs'] = twap_costs
                st.session_state['vwap_costs'] = vwap_costs
                st.session_state['twap_schedules'] = twap_schedules_temp
                st.session_state['vwap_schedules'] = vwap_schedules_temp
                st.session_state['has_baselines_run'] = True
                st.success("Baseline simulations complete!")

                st.markdown(f"### Baseline Performance Summary")
                twap_avg = np.mean(st.session_state['twap_costs'])
                twap_std = np.std(st.session_state['twap_costs'])
                vwap_avg = np.mean(st.session_state['vwap_costs'])
                vwap_std = np.std(st.session_state['vwap_costs'])

                st.markdown(f"TWAP Avg IS: `{twap_avg:.2f} bps` (Std: `{twap_std:.2f} bps`)")
                st.markdown(f"VWAP Avg IS: `{vwap_avg:.2f} bps` (Std: `{vwap_std:.2f} bps`)")

                # V1: Execution Schedule Comparison for one trial
                st.markdown(f"### V1: Execution Schedule Comparison (Sample Trial)")
                fig_schedules, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                ax1.bar(range(st.session_state['base_env_instance'].T), [s[0] for s in st.session_state['twap_schedules'][0]], color='skyblue')
                ax1.set_title('TWAP Schedule')
                ax1.set_xlabel('Period')
                ax1.set_ylabel('Shares Traded')

                ax2.bar(range(st.session_state['base_env_instance'].T), [s[0] for s in st.session_state['vwap_schedules'][0]], color='lightcoral')
                ax2.set_title('VWAP Schedule')
                ax2.set_xlabel('Period')
                ax2.set_ylabel('Shares Traded')

                plt.tight_layout()
                st.pyplot(fig_schedules)
                plt.close(fig_schedules)

                # V2: Cost Distribution for Baselines (using box plot)
                st.markdown(f"### V2: Cost Distribution (Baselines)")
                fig_boxplot, ax = plt.subplots(figsize=(8, 6))
                data_to_plot_baselines = [st.session_state['twap_costs'], st.session_state['vwap_costs']]
                labels_baselines = ['TWAP', 'VWAP']
                ax.boxplot(data_to_plot_baselines, labels=labels_baselines, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='blue'),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='blue'),
                           capprops=dict(color='blue'))
                ax.set_title('Distribution of Implementation Shortfall (Basis Points) - Baselines')
                ax.set_ylabel('Implementation Shortfall (bps)')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig_boxplot)
                plt.close(fig_boxplot)

            except Exception as e:
                st.error(f"Error running baseline simulations: {e}")
                st.session_state['has_baselines_run'] = False
```

---

#### Page: 3. DQN Agent Training

**Markdown**:
```python
st.title("3. Building and Training the Deep Q-Network (DQN) Agent")
st.markdown(f"Sarah understands that truly minimizing execution costs requires an intelligent agent that can adapt in real-time. This is where Reinforcement Learning, specifically a **Deep Q-Network (DQN)**, comes into play. The DQN agent will learn to make decisions (how many shares to trade) by interacting with our simulated market environment, aiming to maximize cumulative future rewards (i.e., minimize total implementation shortfall).")
st.markdown(f"")
st.markdown(f"A DQN consists of several key components:")
st.markdown(f"-   **Q-Network**: A neural network that approximates the optimal action-value function, $Q(s, a)$.")
st.markdown(f"-   **Target Network**: A separate Q-network with periodically updated weights to stabilize training. The target Q-value $Q_{{\text{{target}}}}$ for an action $(s, a)$ is calculated using the Bellman equation:")

st.markdown(r"$$ Q_{{\text{{target}}}}(s, a) = R_{{t+1}} + \gamma \max_{{a'}} Q_{{\text{{target}}}}(s', a') $$")
st.markdown(r"where $R_{{t+1}}$ is the immediate reward, $\gamma$ is the discount factor, and $s'$ is the next state.")

st.markdown(f"-   **Experience Replay Buffer**: A memory that stores past experiences $(s, a, r, s', \text{{done}})$.")
st.markdown(f"-   **Epsilon-Greedy Exploration**: A strategy to balance exploration (trying new actions) and exploitation (using current best policy).")
st.markdown(f"")
st.markdown(f"**Note:** Training an RL agent can be computationally intensive. For a faster demo, one might load pre-trained models. However, to demonstrate the full workflow as defined in `source.py`, we will run the training process.")
```

**Widgets**:
```python
if not st.session_state['has_baselines_run']:
    st.warning("Please configure the environment and run baseline simulations first.")
else:
    n_episodes = st.number_input("Number of Training Episodes", min_value=100, max_value=2000, value=500, step=100)
    batch_size = st.number_input("Batch Size for Replay", min_value=16, max_value=128, value=32, step=16)
    target_update_freq = st.number_input("Target Network Update Frequency (episodes)", min_value=5, max_value=50, value=10, step=5)
```

**Function Invocation / Interaction**:
```python
    if st.button("Train DQN Agent"):
        with st.spinner(f"Training DQN Agent for {n_episodes} episodes... This will take a while."):
            try:
                env_for_training = ExecutionEnvironment(**st.session_state['env_params']) # Create a fresh env for training
                dqn_agent_instance = DQNExecutionAgent(
                    state_dim=env_for_training.observation_space.shape[0],
                    n_actions=env_for_training.action_space.n,
                    action_map=env_for_training.action_map
                )
                dqn_rewards_hist = dqn_agent_instance.train(
                    env_for_training, n_episodes=n_episodes, batch_size=batch_size, target_update_freq=target_update_freq
                )

                st.session_state['dqn_agent'] = dqn_agent_instance
                st.session_state['dqn_rewards_history'] = dqn_rewards_hist
                st.session_state['has_agent_trained'] = True
                st.success("DQN Agent Training Complete!")

                # V1: Learning Curve
                st.markdown(f"### V1: DQN Agent Learning Curve")
                fig_learning_curve, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state['dqn_rewards_history'])
                ax.set_title('DQN Agent Learning Curve (Total Reward per Episode)')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Total Reward (Negative IS in bps)')
                ax.grid(True)
                st.pyplot(fig_learning_curve)
                plt.close(fig_learning_curve)

            except Exception as e:
                st.error(f"Error training DQN Agent: {e}")
                st.session_state['has_agent_trained'] = False
```

---

#### Page: 4. Performance Evaluation

**Markdown**:
```python
st.title("4. Performance Evaluation and Adaptive Behavior Analysis")
st.markdown(f"After training, the critical step for Sarah is to rigorously evaluate the DQN agent. This involves: ")
st.markdown(f"1.  **Quantitative Comparison**: How does the RL agent's execution cost compare statistically to the TWAP and VWAP baselines across many trials?")
st.markdown(f"2.  **Qualitative Adaptive Behavior Analysis**: Does the RL agent actually exhibit intelligent, adaptive behavior? Sarah needs to see if it accelerates execution during favorable price dips or slows down during periods of high market impact or thin liquidity.")
```

**Widgets**:
```python
if not st.session_state['has_agent_trained']:
    st.warning("Please train the DQN Agent first on the 'DQN Agent Training' page.")
else:
    # No additional widgets, uses n_trials from baselines
    st.markdown(f"Evaluation will be run for `{st.session_state['n_trials']}` trials using the trained agent.")
```

**Function Invocation / Interaction**:
```python
    if st.button("Evaluate RL Agent"):
        with st.spinner(f"Evaluating DQN Agent and analyzing adaptive behavior over {st.session_state['n_trials']} trials..."):
            try:
                # Evaluate the trained DQN agent
                rl_costs, rl_schedules_temp = evaluate_agent(st.session_state['dqn_agent'], st.session_state['base_env_instance'], n_trials=st.session_state['n_trials'])
                st.session_state['rl_costs'] = rl_costs
                st.session_state['rl_schedules'] = rl_schedules_temp

                # Analyze adaptive behavior for a typical scenario
                typical_env_params = st.session_state['env_params'].copy()
                typical_env_params['price_drift_mean'] = 0.00005 # Example: slight positive drift for observation
                rl_behavior_df_result = analyze_adaptive_behavior(st.session_state['dqn_agent'], typical_env_params)
                st.session_state['rl_behavior_df'] = rl_behavior_df_result
                st.session_state['has_agent_evaluated'] = True
                st.success("DQN Agent Evaluation Complete!")

                st.markdown(f"### Execution Performance Comparison")
                # Prepare data for display
                strategies_data = {
                    'TWAP': st.session_state['twap_costs'],
                    'VWAP': st.session_state['vwap_costs'],
                    'RL Agent': st.session_state['rl_costs']
                }

                twap_avg = np.mean(st.session_state['twap_costs'])

                performance_df = pd.DataFrame(columns=['Strategy', 'Avg Cost (bps)', 'Std Dev (bps)', 'Savings vs TWAP (bps)'])
                for name, costs in strategies_data.items():
                    avg_cost = np.mean(costs)
                    std_dev = np.std(costs)
                    savings_vs_twap = twap_avg - avg_cost if name != 'TWAP' else 0.0
                    performance_df.loc[len(performance_df)] = [name, avg_cost, std_dev, savings_vs_twap]

                st.dataframe(performance_df.set_index('Strategy').style.format({
                    'Avg Cost (bps)': '{:.2f}',
                    'Std Dev (bps)': '{:.2f}',
                    'Savings vs TWAP (bps)': '{:.2f}'
                }))

                st.markdown(f"### Statistical Significance (RL vs TWAP)")
                t_stat, p_val = ttest_ind(st.session_state['rl_costs'], st.session_state['twap_costs'], equal_var=False)
                st.markdown(f"T-statistic: `{t_stat:.2f}`, P-value: `{p_val:.3f}`")
                if p_val < 0.05:
                    st.markdown(f"The difference in execution costs between RL Agent and TWAP is **statistically significant** (p < 0.05).")
                else:
                    st.markdown(f"The difference in execution costs between RL Agent and TWAP is **NOT statistically significant** (p >= 0.05).")


                # V2: Cost Distribution (all strategies)
                st.markdown(f"### V2: Distribution of Implementation Shortfall (Basis Points)")
                fig_boxplot_all, ax = plt.subplots(figsize=(10, 6))
                data_to_plot_all = [st.session_state['twap_costs'], st.session_state['vwap_costs'], st.session_state['rl_costs']]
                labels_all = ['TWAP', 'VWAP', 'RL Agent']
                ax.boxplot(data_to_plot_all, labels=labels_all, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='blue'),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='blue'),
                           capprops=dict(color='blue'))
                ax.set_title('Distribution of Implementation Shortfall (Basis Points)')
                ax.set_ylabel('Implementation Shortfall (bps)')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig_boxplot_all)
                plt.close(fig_boxplot_all)

                # V3: Example Execution Schedules (TWAP, VWAP, RL) - choose a representative trial
                st.markdown(f"### V3: Execution Schedule Comparison (Sample Trial)")
                fig_schedules_all, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

                ax1.bar(range(st.session_state['base_env_instance'].T), [s[0] for s in st.session_state['twap_schedules'][0]], color='skyblue')
                ax1.set_title('TWAP Schedule')
                ax1.set_xlabel('Period')
                ax1.set_ylabel('Shares Traded')

                ax2.bar(range(st.session_state['base_env_instance'].T), [s[0] for s in st.session_state['vwap_schedules'][0]], color='lightcoral')
                ax2.set_title('VWAP Schedule')
                ax2.set_xlabel('Period')
                ax2.set_ylabel('Shares Traded')

                ax3.bar(range(st.session_state['base_env_instance'].T), [s[0] for s in st.session_state['rl_schedules'][0]], color='lightgreen')
                ax3.set_title('RL Agent Schedule (Example)')
                ax3.set_xlabel('Period')
                ax3.set_ylabel('Shares Traded')

                plt.tight_layout()
                st.pyplot(fig_schedules_all)
                plt.close(fig_schedules_all)

                # V3: Adaptive Behavior - Price path overlaid with trade actions
                st.markdown(f"### V3: RL Agent Adaptive Behavior (Sample Scenario)")
                st.markdown(f"This visualization shows how the RL agent adapts its trading decisions based on market price movements. For instance, it might accelerate execution during favorable price dips or slow down when prices move adversely or market impact is high.")

                fig_adaptive, (ax_price, ax_shares) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                ax_price.plot(st.session_state['rl_behavior_df']['period'], st.session_state['rl_behavior_df']['current_market_price'], label='Market Price', marker='o', linestyle='-')
                ax_price.axhline(y=st.session_state['rl_behavior_df']['current_market_price'].iloc[0], color='r', linestyle='--', label='Initial Price ($P_0$)')
                ax_price.set_title('Market Price Evolution and RL Agent Execution')
                ax_price.set_ylabel('Price')
                ax_price.legend()
                ax_price.grid(True)

                ax_shares.bar(st.session_state['rl_behavior_df']['period'], st.session_state['rl_behavior_df']['shares_traded'], color='purple', alpha=0.7)
                ax_shares.set_title('RL Agent Shares Traded per Period')
                ax_shares.set_xlabel('Period')
                ax_shares.set_ylabel('Shares Traded')
                ax_shares.grid(axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig_adaptive)
                plt.close(fig_adaptive)

                # V4: Implementation Shortfall Breakdown (Conceptual)
                st.markdown(f"### V4: Conceptual Implementation Shortfall Breakdown")
                st.markdown(f"Implementation Shortfall (IS) is generally composed of:")
                st.markdown(f"-   **Market Impact Cost**: Directly due to the agent's trades moving the price (temporary and permanent impact).")
                st.markdown(f"-   **Opportunity Cost (or Timing Cost)**: Due to market price moving away from the decision price while the order is being executed.")
                st.markdown(f"")
                st.markdown(f"While our current simulation environment aggregates these into a total IS, in a real-world analysis, Alpha Capital Partners would use Transaction Cost Analysis (TCA) tools to decompose these costs for each strategy.")
                st.markdown(f"The RL agent aims to dynamically balance these two components: trading faster reduces opportunity cost but increases market impact, and vice-versa.")
                st.markdown(f"For the RL Agent, the average total IS was: `{np.mean(st.session_state['rl_costs']):.2f} bps`")

            except Exception as e:
                st.error(f"Error evaluating DQN Agent: {e}")
                st.session_state['has_agent_evaluated'] = False
```

---

#### Page: 5. Governance & Deployability

**Markdown**:
```python
st.title("5. Towards Deployment: Governance and Risk Management")
st.markdown(f"For Sarah, the quantitative results are compelling, but deploying an autonomous trading agent for large block orders is a significant undertaking that demands rigorous risk management and governance. This is classified as a **Tier 1 autonomous trading model**, meaning it directly makes real-time trading decisions with significant capital at risk.")
st.markdown(f"Alpha Capital Partners must have a robust framework in place to ensure safety, reliability, and human oversight.")
st.markdown(f"")

st.markdown(f"### V5: Governance Dashboard (Conceptual)")
st.markdown(f"")
st.markdown(f"---")
st.markdown(f"### **GOVERNANCE ASSESSMENT: RL EXECUTION AGENT DEPLOYABILITY**")
st.markdown(f"---")

st.markdown(f"Alpha Capital Partners classifies this as a Tier 1 Autonomous Trading Model. This implies the highest level of scrutiny and control.")

st.markdown(f"#### 1. Model Tier & Oversight:")
st.markdown(f"-   **Tier**: 1 (Autonomous Trading Decisions)")
st.markdown(f"-   **Oversight**: Human-on-the-loop (real-time monitoring and intervention capability)")
st.markdown(f"-   **Explainability**: Ability to analyze agent's past decisions (as demonstrated in adaptive behavior analysis).")

st.markdown(f"#### 2. Kill-Switch Criteria (Automated Safeguards):")
st.markdown(f"-   Auto-halt if execution cost (IS) for a specific trade/day exceeds 2x TWAP benchmark.")
st.markdown(f"-   Auto-halt if cumulative execution cost for the RL agent consistently underperforms TWAP for 5+ consecutive days.")
st.markdown(f"-   Auto-halt if market volatility or liquidity deviates significantly from trained parameters (e.g., 3-sigma event).")
st.markdown(f"-   Emergency manual override for human traders/quants.")

st.markdown(f"#### 3. Real-time Monitoring Strategies:")
st.markdown(f"-   Per-trade cost vs. TWAP benchmark (logged and alerted for anomalies).")
st.markdown(f"-   Real-time tracking of remaining shares, time elapsed, and actual vs. target execution schedule.")
st.markdown(f"-   Monitoring of market conditions (volatility, spread, volume) to ensure agent operates within its trained domain.")
st.markdown(f"-   Alerting system for unusual trade sizes or price impact events initiated by the agent.")

st.markdown(f"#### 4. Continuous Validation & Retraining:")
st.markdown(f"-   Daily backtesting of agent's performance against historical order-by-order data before live deployment.")
st.markdown(f"-   Monthly comparison to TWAP/VWAP on live data performance metrics.")
st.markdown(f"-   Quarterly retraining with updated market impact data and evolving market microstructure.")
st.markdown(f"-   A/B testing or shadow trading in a simulated environment before full production rollout.")

st.markdown(f"#### 5. Incident Response Protocol:")
st.markdown(f"-   Defined protocol for investigating cost anomalies or kill-switch activations.")
st.markdown(f"-   Clear rollback procedures and human hand-off mechanisms.")

st.markdown(f"---")
st.markdown(f"### Practitioner Warning: Simulation-to-Production Gap")
st.markdown(f"An RL execution agent trained in simulation may behave differently in production. Our simulated market impact model is a simplification. Real order books have:")
st.markdown(f"-   Discrete tick sizes, not continuous prices.")
st.markdown(f"-   Hidden liquidity and dark pools.")
st.markdown(f"-   Competing algorithms and high-frequency trading.")
st.markdown(f"-   Regime-dependent dynamics (behavior changes in crisis vs. calm markets).")
st.markdown(f"The agent must be rigorously validated on historical order-by-order data before live deployment, and monitored continuously against TWAP/VWAP benchmarks in production to ensure responsible use.")
st.markdown(f"---")
```

**Widgets**: None.
**Function Invocation / Interaction**: No direct function calls, `governance_assessment_discussion` is conceptually integrated into the markdown.
**Output**: Conceptual Governance Dashboard (as detailed markdown).

