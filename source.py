# Standard Library Imports
import random
import warnings
from collections import deque
import os # For checking file existence, though direct download is typically not in app.py

# Third-party Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output in production/app environments
warnings.filterwarnings('ignore')

# --- Global Configuration Parameters ---
# These parameters define the environment and simulation settings.
# They can be adjusted or externalized to a configuration file if needed.
DEFAULT_ENV_PARAMS = {
    'total_shares': 500000,
    'n_periods': 20,
    'initial_price': 150.0,
    'volatility': 0.015,
    'permanent_impact_coeff': 0.00005,
    'temporary_impact_coeff': 0.0005,
    'price_drift_mean': 0.0,
    'price_drift_std': 0.0001
}
N_TRIALS = 100  # Number of simulation trials for statistical comparison
DQN_TRAIN_EPISODES = 80
DQN_BATCH_SIZE = 32
DQN_TARGET_UPDATE_FREQ = 10
DQN_MODEL_PATH = 'dqn_model.keras'
DQN_TARGET_MODEL_PATH = 'dqn_target_model.keras'

# --- 1. Environment Definition ---
class ExecutionEnvironment(gym.Env):
    """
    Almgren-Chriss inspired execution environment.
    Each trade incurs temporary and permanent market impact.
    The environment models market price evolution with drift and noise.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, total_shares=500000, n_periods=20,
                 initial_price=150.0, volatility=0.015,
                 permanent_impact_coeff=0.00005, temporary_impact_coeff=0.0005,
                 price_drift_mean=0.0, price_drift_std=0.0001):
        super().__init__()
        self.Q = total_shares  # Total shares to execute
        self.T = n_periods  # Number of execution periods
        self.P0 = initial_price  # Initial decision price
        self.price = initial_price # Current market price
        self.sigma = volatility  # Market volatility
        self.gamma = permanent_impact_coeff  # Permanent impact per share
        self.eta = temporary_impact_coeff  # Temporary impact per share
        self.price_drift_mean = price_drift_mean
        self.price_drift_std = price_drift_std

        self.remaining_shares = total_shares
        self.current_period = 0
        self.executed_shares = []
        self.execution_prices = []
        self.total_cost = 0.0

        # Define action and observation spaces
        # Action space: discrete action representing a fraction of remaining shares to trade
        # For simplicity, let's discretize into 5 actions for fractions of remaining shares.
        # e.g., [0.05, 0.10, 0.15, 0.25, 0.50] of remaining shares
        self.action_space = spaces.Discrete(5) # Represents indices for action_map

        # Observation space: [fraction_remaining_shares, fraction_time_elapsed, price_drift_from_P0, current_volatility]
        self.observation_space = spaces.Box(
              low=np.array([0., 0., -np.inf, 0.]),
            high=np.array([1., 1., np.inf, np.inf]),
            dtype=np.float32
        )
        # Map discrete actions to actual fractions of remaining shares
        self.action_map = np.array([0.05, 0.10, 0.15, 0.25, 0.50]) # These are fractions of *remaining* shares

    def _get_obs(self):
        """Helper to get current state observation."""
        fraction_remaining = self.remaining_shares / self.Q
        fraction_time_elapsed = self.current_period / self.T
        price_drift_from_P0 = (self.price - self.P0) / self.P0 # Normalized price drift
        # For simplicity, current volatility is assumed constant here, but could be dynamic
        current_volatility = self.sigma
        return np.array([fraction_remaining, fraction_time_elapsed, price_drift_from_P0, current_volatility], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.P0
        self.remaining_shares = self.Q
        self.current_period = 0
        self.executed_shares = []
        self.execution_prices = []
        self.total_cost = 0.0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action_idx):
        if not self.action_space.contains(action_idx):
              raise ValueError(f"Invalid action index: {action_idx}. Must be within {self.action_space.n}")

        action_fraction = self.action_map[action_idx]

        # Calculate shares to trade, ensuring we don't trade more than remaining
        shares_to_trade = int(self.remaining_shares * action_fraction)

        # In the last period, execute all remaining shares
        if self.current_period == self.T - 1:
              shares_to_trade = self.remaining_shares
        else:
            # Ensure at least 1 share if agent wants to trade, to avoid zero trades when action_fraction is small
            # and remaining_shares is also small, unless remaining_shares is 0.
            if shares_to_trade == 0 and self.remaining_shares > 0 and action_fraction > 0:
                   shares_to_trade = 1 # Trade at least 1 share if a positive action fraction is chosen
            shares_to_trade = min(shares_to_trade, self.remaining_shares) # Cannot trade more than remaining

        # Temporary Market Impact: affects current execution price
        temp_impact = self.eta * shares_to_trade
        execution_price = self.price + temp_impact

        # Permanent Market Impact: shifts the market price for future periods
        perm_impact = self.gamma * shares_to_trade
        self.price += perm_impact

        # Market price evolution (stochastic part)
        # Random price fluctuation (drift + noise) for the next period, for unexecuted portions
        market_drift = np.random.normal(loc=self.price_drift_mean, scale=self.price_drift_std) * self.price
        market_noise = np.random.normal(loc=0, scale=self.sigma) * self.price * np.random.randn()
        self.price += market_drift + market_noise # Update price for next observation

        # Update environment state
        self.remaining_shares -= shares_to_trade
        self.current_period += 1
        self.executed_shares.append(shares_to_trade)
        self.execution_prices.append(execution_price)

        # Calculate cost for this step relative to the initial decision price (P0)
        cost_this_step = (execution_price - self.P0) * shares_to_trade
        self.total_cost += cost_this_step

        # Determine if episode is done
        terminated = self.remaining_shares <= 0 or self.current_period >= self.T
        truncated = False # Not using truncation in this specific environment setup

        # Reward: negative implementation shortfall for this trade in basis points
        # Maximizing reward means minimizing cost
        # Total implementation shortfall is what we care about at the end.
        # For intermediate steps, we can use the negative cost of the current trade.
        reward = -(cost_this_step / (self.Q * self.P0)) * 10000 if shares_to_trade > 0 else 0

        observation = self._get_obs()
        info = {
              "current_price": self.price,
            "execution_price": execution_price,
            "shares_traded": shares_to_trade,
            "remaining_shares": self.remaining_shares
        }

        return observation, reward, terminated, truncated, info

# --- 2. Utility Functions ---
def calculate_implementation_shortfall(env_history, Q_total, P0):
    """
    Calculates the implementation shortfall in basis points for a given execution history.
    Args:
        env_history (list): List of (shares_traded, execution_price) tuples for an episode.
        Q_total (int): Initial total shares of the order.
        P0 (float): Initial decision price.
    Returns:
        float: Implementation shortfall in basis points.
    """
    if Q_total == 0 or P0 == 0:
          return 0.0

    total_cost = 0.0
    for shares_traded, exec_price in env_history:
          total_cost += (exec_price - P0) * shares_traded

    # The total_cost accumulated in the environment's step function is sum(n_t * (P_exec,t - P0)).
    return (total_cost / (Q_total * P0)) * 10000

# --- 3. Baseline Strategy Implementations ---
def execute_twap(env_instance):
    """
    Executes a Time-Weighted Average Price (TWAP) strategy.
    Splits the order equally across time.
    Args:
        env_instance (ExecutionEnvironment): An instance of the environment.
    Returns:
        tuple: (final_is_bps, episode_history)
    """
    env = env_instance # Use the provided instance to maintain consistent parameters
    obs, info = env.reset()

    # Shares per period: Ensure integer division and handle remainder for last period
    base_shares_per_period = env.Q // env.T

    episode_history = []

    for t in range(env.T):
        shares_to_trade_this_period = base_shares_per_period
        if t == env.T - 1: # Last period, trade all remaining shares
            shares_to_trade_this_period = env.remaining_shares

        # If remaining shares is 0, then we choose the smallest action or an action that results in 0 shares
        if env.remaining_shares == 0:
              action_idx = 0 # Smallest fraction, results in 0 shares if remaining is 0
        else:
            # Calculate desired fraction of remaining shares to trade for this period's target
            target_fraction = shares_to_trade_this_period / env.remaining_shares
            # Find the closest action index from the predefined action_map
            action_idx = np.argmin(np.abs(env.action_map - target_fraction))

        # Ensure action_idx is within bounds
        action_idx = np.clip(action_idx, 0, env.action_space.n - 1)

        obs, reward, terminated, truncated, info = env.step(action_idx)
        episode_history.append((info['shares_traded'], info['execution_price']))
        if terminated:
              break

    # Calculate total IS from the environment's accumulated cost
    final_is_bps = calculate_implementation_shortfall(episode_history, env.Q, env.P0)
    return final_is_bps, episode_history # Return IS in bps and the full history


def execute_vwap(env_instance, volume_profile=None):
    """
    Executes a Volume-Weighted Average Price (VWAP) strategy.
    Splits the order proportional to expected volume.
    Args:
        env_instance (ExecutionEnvironment): An instance of the environment.
        volume_profile (np.array, optional): Array of volume fractions per period.
                                            If None, a U-shaped profile is generated.
    Returns:
        tuple: (final_is_bps, episode_history)
    """
    env = env_instance
    obs, info = env.reset()

    if volume_profile is None:
        # Generate a conceptual U-shaped volume profile: more volume at start/end
        x = np.linspace(0, 1, env.T)
        volume_profile = 1.5 * (np.abs(x - 0.5) * 2)**2 + 0.5 # More pronounced U-shape
        volume_profile /= volume_profile.sum() # Normalize to 1

    episode_history = []

    for t in range(env.T):
        if env.remaining_shares == 0:
              action_idx = 0 # No shares left to trade
        else:
              # Calculate target shares for this period based on volume profile
            target_shares_this_period = env.Q * volume_profile[t]
            # Calculate desired fraction of remaining shares to trade
            target_fraction = target_shares_this_period / env.remaining_shares
            # Find the closest action index from the predefined action_map
            action_idx = np.argmin(np.abs(env.action_map - target_fraction))

        # Ensure action_idx is within bounds
        action_idx = np.clip(action_idx, 0, env.action_space.n - 1)

        obs, reward, terminated, truncated, info = env.step(action_idx)
        episode_history.append((info['shares_traded'], info['execution_price']))
        if terminated:
              break

    final_is_bps = calculate_implementation_shortfall(episode_history, env.Q, env.P0)
    return final_is_bps, episode_history

# --- 4. DQN Agent Definition ---
class DQNExecutionAgent:
    """
    Deep Q-Network (DQN) agent for adaptive trade execution.
    """
    def __init__(self, state_dim, n_actions, action_map):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.action_map = action_map # Actual fractional actions

        self.memory = deque(maxlen=20000) # Experience replay buffer
        self.gamma = 0.99    # Discount factor for future rewards
        self.epsilon = 1.0   # Exploration-exploitation trade-off parameter
        self.epsilon_min = 0.05 # Minimum epsilon value
        self.epsilon_decay = 0.997 # Rate of epsilon decay

        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds the Keras neural network model for the Q-function."""
        model = keras.Sequential([
              keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action based on epsilon-greedy strategy.
        State needs to be reshaped for model prediction.
        """
        if np.random.rand() <= self.epsilon:
              return random.randrange(self.n_actions) # Explore: choose random action index

        # Exploit: choose action with highest Q-value
        # state is expected to be a numpy array, reshape for model.predict
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values) # Return action index

    def replay(self, batch_size):
        """
        Trains the agent using a mini-batch of experiences from the replay buffer.
        """
        if len(self.memory) < batch_size:
              return

        minibatch = random.sample(self.memory, batch_size)

        # Extract states, actions, rewards, next_states, and done flags
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # Predict Q-values for current states using the main model
        current_q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next states using the target model
        target_q_values_next_state = self.target_model.predict(next_states, verbose=0)

        # Create target Q-values for training
        targets = np.copy(current_q_values)

        for i in range(batch_size):
            if dones[i]:
                # If episode is done, target Q-value is just the immediate reward
                targets[i][actions[i]] = rewards[i]
            else:
                # Bellman equation: Target Q-value = Reward + Gamma * max(Q_target(s', a'))
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(target_q_values_next_state[i])

        # Train the main model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
              self.epsilon *= self.epsilon_decay

    def train(self, env_instance, n_episodes=100, batch_size=32, target_update_freq=10):
        """
        Trains the DQN agent in the given environment for a number of episodes.
        """
        rewards_history = []
        env = env_instance # Use the environment instance passed

        print(f"Starting DQN training for {n_episodes} episodes...")
        for ep in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False

            while not done:
                action_idx = self.act(state) # Agent chooses action (index)

                next_state, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated

                self.remember(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.replay(batch_size) # Train the agent

            rewards_history.append(total_reward)

            # Update target network periodically
            if (ep + 1) % target_update_freq == 0:
                  self.update_target_model()

            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-min(50, len(rewards_history)):]) # Average over last 50 episodes
                print(f"Episode {ep+1}/{n_episodes}: Avg Reward={avg_reward:.2f}, Epsilon={self.epsilon:.3f}")

        return rewards_history

# --- 5. Orchestration and Analysis Functions ---

def load_dqn_models(env_template, model_path, target_model_path):
    """
    Loads pre-trained DQN models and initializes the agent with them.
    If models are not found, a new, untrained agent is created.
    Args:
        env_template (ExecutionEnvironment): An instance of ExecutionEnvironment
                                             to get state/action dims.
        model_path (str): Path to the main DQN model file.
        target_model_path (str): Path to the target DQN model file.
    Returns:
        DQNExecutionAgent: An initialized DQN agent with loaded models (or new if not found).
    """
    state_dim = env_template.observation_space.shape[0]
    n_actions = env_template.action_space.n
    action_map = env_template.action_map

    if os.path.exists(model_path) and os.path.exists(target_model_path):
        try:
            main_model = keras.models.load_model(model_path)
            target_model = keras.models.load_model(target_model_path)
            print("DQN models loaded successfully.")

            agent = DQNExecutionAgent(state_dim, n_actions, action_map)
            agent.model = main_model
            agent.target_model = target_model
            return agent
        except Exception as e:
            print(f"Error loading DQN models: {e}. Creating a new, untrained agent.")
            return DQNExecutionAgent(state_dim, n_actions, action_map)
    else:
        print(f"DQN models not found at '{model_path}' and '{target_model_path}'. Creating a new, untrained agent.")
        return DQNExecutionAgent(state_dim, n_actions, action_map)

def train_dqn_agent(agent, env_params, n_episodes, batch_size, target_update_freq, plot_curve=False):
    """
    Trains the DQN agent and returns its reward history.
    Args:
        agent (DQNExecutionAgent): The DQN agent instance to train.
        env_params (dict): Parameters for the ExecutionEnvironment.
        n_episodes (int): Number of episodes for training.
        batch_size (int): Batch size for experience replay.
        target_update_freq (int): Frequency to update the target model.
        plot_curve (bool): If True, plots the training reward curve.
    Returns:
        list: History of total rewards per episode.
    """
    print("\nStarting DQN Agent Training...")
    train_env = ExecutionEnvironment(**env_params) # Create a fresh environment for training
    rewards_history = agent.train(train_env, n_episodes, batch_size, target_update_freq)
    print("DQN Agent Training Complete.")

    if plot_curve:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history)
        plt.title('DQN Agent Learning Curve (Total Reward per Episode)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward (Negative IS in bps)')
        plt.grid(True)
        plt.show()
    return rewards_history

def evaluate_agent(agent, env_params, n_trials=10):
    """
    Evaluates the performance of a trained agent over multiple trials.
    Returns a list of implementation shortfall costs in basis points and schedules.
    Args:
        agent (DQNExecutionAgent): The trained DQN agent instance.
        env_params (dict): Parameters for the ExecutionEnvironment.
        n_trials (int): Number of evaluation trials.
    Returns:
        tuple: (list of costs in bps, list of execution schedules)
    """
    costs = []
    agent_schedules = []
    env_instance = ExecutionEnvironment(**env_params) # Create a fresh environment for evaluation

    print(f"\nEvaluating Agent performance over {n_trials} trials...")
    # Temporarily set epsilon to 0 for evaluation to ensure greedy actions
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for i in range(n_trials):
        state, info = env_instance.reset()
        done = False
        episode_history = []

        while not done:
            q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
            action_idx = np.argmax(q_values) # Choose action with highest Q-value

            next_state, reward, terminated, truncated, info = env_instance.step(action_idx)
            done = terminated or truncated
            state = next_state
            episode_history.append((info['shares_traded'], info['execution_price']))

        final_is_bps = calculate_implementation_shortfall(episode_history, env_instance.Q, env_instance.P0)
        costs.append(final_is_bps)
        agent_schedules.append(episode_history)
        if (i + 1) % (n_trials // 10 or 1) == 0:
            print(f"  Completed {i+1}/{n_trials} evaluation trials.")

    print("Agent Evaluation Complete.")
    agent.epsilon = original_epsilon # Restore original epsilon
    return costs, agent_schedules

def run_baseline_simulations(env_params, n_trials):
    """
    Runs TWAP and VWAP simulations and returns their costs and schedules.
    Args:
        env_params (dict): Parameters for the ExecutionEnvironment.
        n_trials (int): Number of simulation trials.
    Returns:
        tuple: (list of TWAP costs, list of VWAP costs,
                list of TWAP schedules, list of VWAP schedules)
    """
    print(f"\nRunning baseline simulations (TWAP, VWAP) over {n_trials} trials...")
    twap_costs = []
    vwap_costs = []
    twap_schedules = []
    vwap_schedules = []

    # Instantiate a single environment template for baselines; each function will reset it internally
    base_env = ExecutionEnvironment(**env_params)

    for i in range(n_trials):
        cost, schedule = execute_twap(base_env)
        twap_costs.append(cost)
        twap_schedules.append(schedule)

        cost, schedule = execute_vwap(base_env)
        vwap_costs.append(cost)
        vwap_schedules.append(schedule)
        if (i + 1) % (n_trials // 10 or 1) == 0:
            print(f"  Completed {i+1}/{n_trials} baseline trials.")

    print("Baseline simulations complete.")
    return twap_costs, vwap_costs, twap_schedules, vwap_schedules

def compare_performance(twap_costs, vwap_costs, rl_costs):
    """
    Prints a comparison table of execution costs and performs statistical tests.
    Args:
        twap_costs (list): List of TWAP implementation shortfall costs.
        vwap_costs (list): List of VWAP implementation shortfall costs.
        rl_costs (list): List of RL agent implementation shortfall costs.
    """
    print("\nEXECUTION PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<20s}{'Avg Cost (bps)':>18s}{'Std Dev (bps)':>18s}{'Savings vs TWAP (bps)':>22s}")
    print("-" * 70)

    strategies = {
        'TWAP': twap_costs,
        'VWAP': vwap_costs,
        'RL Agent': rl_costs
    }

    twap_avg = np.mean(twap_costs) if twap_costs else 0.0

    for name, costs in strategies.items():
        if not costs: # Handle empty lists if a strategy wasn't run or failed
            avg_cost = 0.0
            std_dev = 0.0
            savings_vs_twap = 0.0
        else:
            avg_cost = np.mean(costs)
            std_dev = np.std(costs)
            savings_vs_twap = twap_avg - avg_cost if name != 'TWAP' else 0.0
        print(f"{name:<20s}{avg_cost:>18.2f}{std_dev:>18.2f}{savings_vs_twap:>22.2f}")

    if rl_costs and twap_costs and len(rl_costs) > 1 and len(twap_costs) > 1:
        print("\nStatistical Significance (RL vs TWAP):")
        t_stat, p_val = ttest_ind(rl_costs, twap_costs, equal_var=False) # Welch's t-test
        print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.3f}")
        if p_val < 0.05:
            print("The difference in execution costs between RL Agent and TWAP is statistically significant (p < 0.05).")
        else:
            print("The difference in execution costs between RL Agent and TWAP is NOT statistically significant (p >= 0.05).")
    else:
        print("\nNot enough data to perform statistical significance test for RL vs TWAP.")

def plot_execution_results(twap_costs, vwap_costs, rl_costs, twap_schedules, vwap_schedules, rl_schedules, env_params):
    """
    Generates various plots for performance comparison and schedule visualization.
    Args:
        twap_costs (list): List of TWAP implementation shortfall costs.
        vwap_costs (list): List of VWAP implementation shortfall costs.
        rl_costs (list): List of RL agent implementation shortfall costs.
        twap_schedules (list): List of TWAP execution schedules.
        vwap_schedules (list): List of VWAP execution schedules.
        rl_schedules (list): List of RL agent execution schedules.
        env_params (dict): Parameters for the ExecutionEnvironment.
    """
    base_env_T = env_params.get('n_periods', 20)

    # V1: Box plot of execution costs
    data_to_plot = [costs for costs in [twap_costs, vwap_costs, rl_costs] if costs] # Filter out empty lists
    labels = [label for label, costs in zip(['TWAP', 'VWAP', 'RL Agent'], [twap_costs, vwap_costs, rl_costs]) if costs]

    if data_to_plot:
        plt.figure(figsize=(10, 6))
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'))
        plt.title('Distribution of Implementation Shortfall (Basis Points)')
        plt.ylabel('Implementation Shortfall (bps)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to generate Box Plot of execution costs.")

    # V2: Example Execution Schedules (TWAP, VWAP, RL) - choose a representative trial (e.g., trial 0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if twap_schedules:
        axes[0].bar(range(base_env_T), [s[0] for s in twap_schedules[0]], color='skyblue')
        axes[0].set_title('TWAP Schedule')
        axes[0].set_xlabel('Period')
        axes[0].set_ylabel('Shares Traded')
    else:
        axes[0].set_visible(False) # Hide subplot if no data

    if vwap_schedules:
        axes[1].bar(range(base_env_T), [s[0] for s in vwap_schedules[0]], color='lightcoral')
        axes[1].set_title('VWAP Schedule')
        axes[1].set_xlabel('Period')
        axes[1].set_ylabel('Shares Traded')
    else:
        axes[1].set_visible(False)

    if rl_schedules:
        axes[2].bar(range(base_env_T), [s[0] for s in rl_schedules[0]], color='lightgreen')
        axes[2].set_title('RL Agent Schedule (Example)')
        axes[2].set_xlabel('Period')
        axes[2].set_ylabel('Shares Traded')
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    plt.show()

def analyze_adaptive_behavior(agent, env_params):
    """
    Analyzes the RL agent's adaptive execution behavior under specific conditions.
    Args:
        agent (DQNExecutionAgent): The trained DQN agent instance.
        env_params (dict): Parameters for the ExecutionEnvironment to simulate a scenario.
    Returns:
        pd.DataFrame: A trace of the agent's decisions and market conditions.
    """
    env = ExecutionEnvironment(**env_params)
    state, info = env.reset()

    execution_trace = []
    # Set epsilon to 0 for analysis to ensure deterministic greedy action
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for t in range(env.T):
        current_price = env.price

        q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        action_idx = np.argmax(q_values)
        action_fraction = agent.action_map[action_idx]

        shares_to_trade_this_step = int(env.remaining_shares * action_fraction)
        if t == env.T - 1:
              shares_to_trade_this_step = env.remaining_shares
        else:
            # Ensure at least 1 share if agent wants to trade, to avoid zero trades when action_fraction is small
            # and remaining_shares is also small, unless remaining_shares is 0.
            if shares_to_trade_this_step == 0 and env.remaining_shares > 0 and action_fraction > 0:
                   shares_to_trade_this_step = 1
            shares_to_trade_this_step = min(shares_to_trade_this_step, env.remaining_shares) # Cannot trade more than remaining


        execution_trace.append({
              'period': t,
            'current_market_price': current_price,
            'price_drift_from_P0': (current_price - env.P0) / env.P0 * 100, # In percentage
            'shares_remaining': env.remaining_shares,
            'chosen_action_fraction': action_fraction,
            'shares_traded': shares_to_trade_this_step
        })

        state, reward, terminated, truncated, info = env.step(action_idx)
        if terminated or truncated:
              break

    # Restore original epsilon
    agent.epsilon = original_epsilon

    df_trace = pd.DataFrame(execution_trace)
    return df_trace

def plot_adaptive_behavior(rl_behavior_df, initial_price):
    """
    Plots the RL agent's adaptive behavior (market price and shares traded).
    Args:
        rl_behavior_df (pd.DataFrame): DataFrame containing the agent's execution trace.
        initial_price (float): The initial price (P0) from the environment.
    """
    if rl_behavior_df.empty:
        print("No RL agent behavior data to plot.")
        return

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(rl_behavior_df['period'], rl_behavior_df['current_market_price'], label='Market Price', marker='o', linestyle='-')
    plt.axhline(y=initial_price, color='r', linestyle='--', label='Initial Price ($P_0$)')
    plt.title('Market Price Evolution and RL Agent Execution')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.bar(rl_behavior_df['period'], rl_behavior_df['shares_traded'], color='purple', alpha=0.7)
    plt.title('RL Agent Shares Traded per Period')
    plt.xlabel('Period')
    plt.ylabel('Shares Traded')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def governance_assessment_discussion():
    """
    Conceptual discussion points on governance, monitoring, and validation
    for a production-ready RL trading agent.
    """
    print("\n" + "="*80)
    print("GOVERNANCE ASSESSMENT: RL EXECUTION AGENT DEPLOYABILITY")
    print("="*80)

    print("\nAlpha Capital Partners classifies this as a Tier 1 Autonomous Trading Model.")
    print("This implies the highest level of scrutiny and control.")

    print("\n1. Model Tier & Oversight:")
    print("   - Tier: 1 (Autonomous Trading Decisions)")
    print("   - Oversight: Human-on-the-loop (real-time monitoring and intervention capability)")
    print("   - Explainability: Ability to analyze agent's past decisions (as demonstrated in adaptive behavior analysis).")

    print("\n2. Kill-Switch Criteria (Automated Safeguards):")
    print("   - Auto-halt if execution cost (IS) for a specific trade/day exceeds 2x TWAP benchmark.")
    print("   - Auto-halt if cumulative execution cost for the RL agent consistently underperforms TWAP for 5+ consecutive days.")
    print("   - Auto-halt if market volatility or liquidity deviates significantly from trained parameters (e.g., 3-sigma event).")
    print("   - Emergency manual override for human traders/quants.")

    print("\n3. Real-time Monitoring Strategies:")
    print("   - Per-trade cost vs. TWAP benchmark (logged and alerted for anomalies).")
    print("   - Real-time tracking of remaining shares, time elapsed, and actual vs. target execution schedule.")
    print("   - Monitoring of market conditions (volatility, spread, volume) to ensure agent operates within its trained domain.")
    print("   - Alerting system for unusual trade sizes or price impact events initiated by the agent.")

    print("\n4. Continuous Validation & Retraining:")
    print("   - Daily backtesting of agent's performance against historical order-by-order data before live deployment.")
    print("   - Monthly comparison to TWAP/VWAP on live data performance metrics.")
    print("   - Quarterly retraining with updated market impact data and evolving market microstructure.")
    print("   - A/B testing or shadow trading in a simulated environment before full production rollout.")

    print("\n5. Incident Response Protocol:")
    print("   - Defined protocol for investigating cost anomalies or kill-switch activations.")
    print("   - Clear rollback procedures and human hand-off mechanisms.")

    print("\n--- Practitioner Warning: Simulation-to-Production Gap ---")
    print("An RL execution agent trained in simulation may behave differently in production.")
    print("Our simulated market impact model is a simplification. Real order books have:")
    print("   - Discrete tick sizes, not continuous prices.")
    print("   - Hidden liquidity and dark pools.")
    print("   - Competing algorithms and high-frequency trading.")
    print("   - Regime-dependent dynamics (behavior changes in crisis vs. calm markets).")
    print("The agent must be rigorously validated on historical order-by-order data before live deployment,")
    print("and monitored continuously against TWAP/VWAP benchmarks in production to ensure responsible use.")
    print("="*80)

# --- Main function to orchestrate the entire process ---
def run_execution_analysis(env_params=None, n_trials=N_TRIALS,
                          train_dqn_flag=False, dqn_episodes=DQN_TRAIN_EPISODES,
                          dqn_batch_size=DQN_BATCH_SIZE, dqn_target_update_freq=DQN_TARGET_UPDATE_FREQ,
                          plot_training_curve=False,
                          dqn_model_path=DQN_MODEL_PATH, dqn_target_model_path=DQN_TARGET_MODEL_PATH):
    """
    Orchestrates the entire trade execution simulation and analysis.
    This function can be imported and called from an app.py file.

    Args:
        env_params (dict, optional): Dictionary of environment parameters.
                                     Defaults to DEFAULT_ENV_PARAMS.
        n_trials (int): Number of simulation trials for each strategy.
        train_dqn_flag (bool): If True, trains a new DQN agent. Otherwise, loads pre-trained.
        dqn_episodes (int): Number of episodes for DQN training.
        dqn_batch_size (int): Batch size for DQN experience replay.
        dqn_target_update_freq (int): Frequency for updating DQN target model.
        plot_training_curve (bool): If True, displays the DQN training reward curve.
        dqn_model_path (str): Path to save/load the main DQN model.
        dqn_target_model_path (str): Path to save/load the target DQN model.

    Returns:
        dict: A dictionary containing simulation results and analysis data.
    """
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS

    print("Initializing environment and agent for analysis...")
    # Create a template environment to get state/action space dimensions for agent initialization
    base_env_for_dims = ExecutionEnvironment(**env_params)

    # Load or create DQN agent
    dqn_agent = load_dqn_models(base_env_for_dims, dqn_model_path, dqn_target_model_path)
    
    if train_dqn_flag:
        _ = train_dqn_agent(dqn_agent, env_params, dqn_episodes, dqn_batch_size, dqn_target_update_freq, plot_training_curve)
        # Save the newly trained models
        try:
            dqn_agent.model.save(dqn_model_path)
            dqn_agent.target_model.save(dqn_target_model_path)
            print(f"Newly trained DQN models saved to {dqn_model_path} and {dqn_target_model_path}.")
        except Exception as e:
            print(f"Failed to save DQN models: {e}")

    # Run simulations for all strategies
    twap_costs, vwap_costs, twap_schedules, vwap_schedules = run_baseline_simulations(env_params, n_trials)
    rl_costs, rl_schedules = evaluate_agent(dqn_agent, env_params, n_trials)

    # Performance Comparison
    compare_performance(twap_costs, vwap_costs, rl_costs)

    # Visualizations
    plot_execution_results(twap_costs, vwap_costs, rl_costs, twap_schedules, vwap_schedules, rl_schedules, env_params)

    # Adaptive Behavior Analysis for a typical scenario
    typical_env_params_for_analysis = env_params.copy()
    typical_env_params_for_analysis['price_drift_mean'] = 0.00005 # Introduce a slight positive drift for observation
    rl_behavior_df = analyze_adaptive_behavior(dqn_agent, typical_env_params_for_analysis)
    print("\nRL Agent Adaptive Behavior Trace (First 5 periods):")
    print(rl_behavior_df.head())
    plot_adaptive_behavior(rl_behavior_df, typical_env_params_for_analysis['initial_price'])

    # Conceptual discussion
    print("\n--- Conceptual Implementation Shortfall Breakdown ---")
    print("Implementation Shortfall (IS) is generally composed of:")
    print("1. Market Impact Cost: Directly due to the agent's trades moving the price (temporary and permanent impact).")
    print("2. Opportunity Cost (or Timing Cost): Due to market price moving away from the decision price while the order is being executed.")
    print("\nWhile our current simulation environment aggregates these into a total IS, in a real-world analysis,")
    print("Alpha Capital Partners would use transaction cost analysis (TCA) tools to decompose these costs for each strategy.")
    print("The RL agent aims to dynamically balance these two components: trading faster reduces opportunity cost but increases market impact, and vice-versa.")
    print(f"For the RL Agent, the average total IS was: {np.mean(rl_costs):.2f} bps")

    governance_assessment_discussion()

    return {
        "twap_costs": twap_costs,
        "vwap_costs": vwap_costs,
        "rl_costs": rl_costs,
        "twap_avg_is": np.mean(twap_costs) if twap_costs else 0.0,
        "vwap_avg_is": np.mean(vwap_costs) if vwap_costs else 0.0,
        "rl_avg_is": np.mean(rl_costs) if rl_costs else 0.0,
        "rl_behavior_trace": rl_behavior_df.to_dict('records') # Return data for app.py
    }

# Entry point for the script if run directly
if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Gymnasium Version: {gym.__version__}")

    # Note: In a production app.py, you would ensure models are pre-deployed
    # or implement a robust download/check mechanism.
    # The original notebook used !wget, which is shell specific.
    # For direct execution, you might need to manually download or
    # implement Pythonic download logic if models are not present.
    # Example (uncomment to enable download if models are missing):
    # from urllib.request import urlretrieve
    # if not (os.path.exists(DQN_MODEL_PATH) and os.path.exists(DQN_TARGET_MODEL_PATH)):
    #     print("Downloading pre-trained models...")
    #     try:
    #         urlretrieve("https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+58/dqn_model.keras", DQN_MODEL_PATH)
    #         urlretrieve("https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+58/dqn_target_model.keras", DQN_TARGET_MODEL_PATH)
    #         print("Models downloaded.")
    #     except Exception as e:
    #         print(f"Failed to download models: {e}. Please ensure you have internet access or download them manually.")
    #         print("Proceeding with untrained agent if models are not found locally.")

    # Call the main analysis function. Set train_dqn_flag=True to train a new agent.
    # Otherwise, it will attempt to load existing models.
    results = run_execution_analysis(train_dqn_flag=False, n_trials=N_TRIALS)
    print("\nSimulation and Analysis Complete. Results returned by run_execution_analysis function.")
