# Databricks notebook source
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Generative Claim Model (Variational Autoencoder)
class GenerativeClaimModel:
    def __init__(self, input_dim=1, latent_dim=2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.encoder, self.decoder = self.build_vae()

    def build_vae(self):
        # Encoder
        input_data = Input(shape=(self.input_dim,), name="encoder_input")
        x = Dense(64, activation="relu", name="encoder_hidden")(input_data)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        # Sampling Layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, name="z")([z_mean, z_log_var])
        encoder = Model(inputs=input_data, outputs=[z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_input = Input(shape=(self.latent_dim,), name="decoder_input")
        x = Dense(64, activation="relu", name="decoder_hidden")(latent_input)
        reconstructed = Dense(self.input_dim, activation="sigmoid", name="decoder_output")(x)
        decoder = Model(inputs=latent_input, outputs=reconstructed, name="decoder")

        return encoder, decoder

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        # Reconstruction Loss
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=1))
        # KL Divergence Loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        return reconstruction_loss + kl_loss

    def train(self, data, epochs=5, batch_size=16, validation_split=0.1):
        optimizer = Adam(learning_rate=0.001)

        # Training Loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                with tf.GradientTape() as tape:
                    # Forward Pass
                    z_mean, z_log_var, z = self.encoder(batch)
                    outputs = self.decoder(z)
                    # Compute Loss
                    loss = self.vae_loss(batch, outputs, z_mean, z_log_var)

                # Backward Pass and Optimization
                gradients = tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.encoder.trainable_weights + self.decoder.trainable_weights))
            print(f"Loss: {loss.numpy():.4f}")

    def generate_claim(self, n_samples=1):
        latent_samples = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decoder.predict(latent_samples)

# Insurance Environment
class InsurerEnv(gym.Env):
    def __init__(self, T, n, lambda_, alpha_bounds, num_layers, budget_max, ruin_prob_target, generative_model):
        super(InsurerEnv, self).__init__()
        self.T = T
        self.n = n
        self.dt = T / n
        self.lambda_ = lambda_
        self.alpha_bounds = alpha_bounds
        self.num_layers = num_layers
        self.budget_max = budget_max
        self.ruin_prob_target = ruin_prob_target
        self.generative_model = generative_model

        # Initialize environment parameters
        self.layer_bounds = [(0, 50)] * self.num_layers
        self.alpha_k = [0.5] * self.num_layers
        self.premium_rate = 4.0 * self.lambda_
        self.reset()

        # Observation and action spaces
        obs_dim = 2 + self.num_layers + 2 * self.num_layers
        action_dim = self.num_layers + 2 * self.num_layers
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def step(self, action):
        self._apply_action(action)
        claims = self._simulate_claims()
        total_retained_loss = self._calculate_retained_loss(claims)

        self.surplus += self.premium_rate * self.dt - total_retained_loss
        reward = self._calculate_reward(total_retained_loss)

        self.current_time += self.dt
        done = self.surplus < -100 or self.current_time >= self.T

        info = {"surplus": self.surplus, "retained_loss": total_retained_loss, "time": self.current_time, "claims": claims}
        return self._get_state(), reward, done, info

    def _apply_action(self, action):
        self.alpha_k = np.clip(
            np.array(self.alpha_k) + action[:self.num_layers],
            self.alpha_bounds[0],
            self.alpha_bounds[1],
        )
        delta_bounds = action[self.num_layers:].reshape(self.num_layers, 2)
        self.layer_bounds = [
            (
                max(0, self.layer_bounds[i][0] + delta_bounds[i, 0]),
                max(self.layer_bounds[i][0], self.layer_bounds[i][1] + delta_bounds[i, 1]),
            )
            for i in range(self.num_layers)
        ]

    def _simulate_claims(self):
        return np.random.poisson(self.lambda_ * self.dt)

    def _calculate_retained_loss(self, claims):
        total_retained_loss = 0
        for _ in range(claims):
            claim_size = self.generative_model.generate_claim(n_samples=1)[0][0] * 100
            retained_loss = 0
            for i in range(self.num_layers):
                a_k, b_k = self.layer_bounds[i]
                layer_loss = np.clip(claim_size - a_k, 0, b_k - a_k)
                retained_loss += self.alpha_k[i] * layer_loss
            total_retained_loss += retained_loss
        return total_retained_loss

    def _calculate_reward(self, total_retained_loss):
        surplus_reward = np.log(max(self.surplus, 1e-5)) if self.surplus > 0 else -1e6
        loss_penalty = -total_retained_loss / (self.premium_rate * self.dt)
        stability_penalty = -np.var([b - a for a, b in self.layer_bounds])
        ruin_penalty = -1e6 if self.surplus < -100 else 0

        return surplus_reward + 0.5 * loss_penalty + 0.1 * stability_penalty + ruin_penalty

    def _get_state(self):
        flattened_bounds = [val for bounds in self.layer_bounds for val in bounds]
        state = np.concatenate(([self.surplus, self.lambda_], self.alpha_k, flattened_bounds))
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def reset(self):
        self.surplus = 20000.0
        self.alpha_k = [0.3] * self.num_layers
        self.layer_bounds = [(0, 50)] * self.num_layers
        self.premium_rate = 4.0 * self.lambda_
        self.current_time = 0.0
        return self._get_state()

# Logging Callback
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.surplus_log = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "surplus" in info:
                    self.surplus_log.append(info["surplus"])
        return True

    def _on_training_end(self) -> None:
        plt.plot(self.surplus_log, label="Surplus Over Time")
        plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
        plt.xlabel("Steps")
        plt.ylabel("Surplus")
        plt.legend()
        plt.show()

# Main Program
if __name__ == "__main__":
    # Generate Training Data
    data = np.random.lognormal(mean=3.5, sigma=1.0, size=(5000, 1)) / 100
    generative_model = GenerativeClaimModel(input_dim=1, latent_dim=2)
    generative_model.train(data, epochs=5, batch_size=16)

    # Create Environment
    scenario = {
        "T": 10,
        "n": 200,
        "lambda_": 10,
        "alpha_bounds": (0.2, 0.5),
        "num_layers": 5,
        "budget_max": 150_000,
        "ruin_prob_target": 0.01,
        "generative_model": generative_model,
    }
    env = InsurerEnv(**scenario)

    # Train RL Agent
    callback = LoggingCallback()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000, callback=callback)


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Out-of-Sample Testing Function
def out_of_sample_testing(env, generative_model, mu, sigma, num_steps=200):
    """
    Perform out-of-sample testing on the environment.

    Parameters:
        env (InsurerEnv): The insurance environment.
        generative_model (GenerativeClaimModel): The trained generative model.
        mu (float): Mean of the out-of-sample claim distribution.
        sigma (float): Standard deviation of the out-of-sample claim distribution.
        num_steps (int): Number of steps to simulate.

    Returns:
        dict: Metrics including mean surplus and ruin probability.
    """
    env.reset()
    surplus_history = []
    ruin_count = 0

    for step in range(num_steps):
        # Generate claims from out-of-sample distribution
        claims = np.random.lognormal(mean=mu, sigma=sigma, size=(1, 1)) / 100

        # Simulate the claims using the generative model
        generated_claim = generative_model.generate_claim(n_samples=1)[0][0] * 100

        # Use a random policy for simplicity (replace with a trained RL policy if available)
        action = env.action_space.sample()

        # Step the environment
        state, reward, done, info = env.step(action)
        surplus_history.append(info["surplus"])

        # Track ruin events
        if info["surplus"] < -100:
            ruin_count += 1

        if done:
            break

    # Calculate metrics
    mean_surplus = np.mean(surplus_history)
    ruin_probability = ruin_count / num_steps

    return {
        "mean_surplus": mean_surplus,
        "ruin_probability": ruin_probability,
        "surplus_history": surplus_history
    }

# Generate Out-of-Sample Dataset
mu_out_of_sample = 3.7
sigma_out_of_sample = 1.1

# Perform Out-of-Sample Testing
metrics = out_of_sample_testing(env, generative_model, mu=mu_out_of_sample, sigma=sigma_out_of_sample, num_steps=500)

# Plot Surplus Over Time
plt.figure(figsize=(10, 6))
plt.plot(metrics["surplus_history"], label="Surplus Over Time")
plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
plt.xlabel("Steps", fontsize=14)
plt.ylabel("Surplus", fontsize=14)
plt.title("Out-of-Sample Testing: Surplus Dynamics", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("out_of_sample_testing.png")
plt.show()

# Print Metrics
print("Out-of-Sample Testing Results:")
print(f"Mean Surplus: {metrics['mean_surplus']:.2f}")
print(f"Ruin Probability: {metrics['ruin_probability']:.2%}")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Enhanced Visualization and Additional Metrics Functions
def plot_surplus_distribution(surplus_history):
    """
    Plot a histogram of surplus values to assess variability.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(surplus_history, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Surplus Distribution", fontsize=16)
    plt.xlabel("Surplus", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.savefig("surplus_distribution.png")
    plt.show()

def plot_claim_size_distribution(training_claims, out_of_sample_claims):
    """
    Plot training and out-of-sample claim size distributions for comparison.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(training_claims, bins=30, alpha=0.7, label="Training Claims", color='blue', edgecolor='k')
    plt.hist(out_of_sample_claims, bins=30, alpha=0.7, label="Out-of-Sample Claims", color='green', edgecolor='k')
    plt.title("Claim Size Distribution", fontsize=16)
    plt.xlabel("Claim Size", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("claim_size_distribution.png")
    plt.show()

def plot_ruin_events(surplus_history):
    """
    Plot surplus over time, marking potential ruin points.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(surplus_history, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.title("Ruin Events Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Surplus", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("ruin_events.png")
    plt.show()

def compute_metrics(surplus_history, total_retained_losses, total_claims):
    """
    Compute additional metrics: variance of surplus and claim coverage efficiency.

    Parameters:
        surplus_history (list): History of surplus values.
        total_retained_losses (float): Sum of all retained losses.
        total_claims (float): Sum of all claims.

    Returns:
        dict: Metrics including variance of surplus and claim coverage efficiency.
    """
    variance_surplus = np.var(surplus_history)
    coverage_efficiency = total_retained_losses / total_claims if total_claims > 0 else 0

    return {
        "variance_surplus": variance_surplus,
        "coverage_efficiency": coverage_efficiency
    }

def sensitivity_analysis(env, generative_model, mu_values, sigma_values, num_steps=200):
    """
    Perform sensitivity analysis by tweaking μ and σ.

    Parameters:
        env (InsurerEnv): The insurance environment.
        generative_model (GenerativeClaimModel): The trained generative model.
        mu_values (list): List of μ values to test.
        sigma_values (list): List of σ values to test.
        num_steps (int): Number of steps to simulate for each configuration.

    Returns:
        dict: Sensitivity results for each configuration.
    """
    results = {}

    for mu in mu_values:
        for sigma in sigma_values:
            metrics = out_of_sample_testing(env, generative_model, mu=mu, sigma=sigma, num_steps=num_steps)
            results[(mu, sigma)] = metrics

    return results

# Fixing issue with numpy.float64 in retained loss calculation
def calculate_retained_loss(env, claim):
    """
    Calculate the retained loss for a single claim.
    """
    total_retained_loss = 0
    claim_size = claim * 100
    retained_loss = 0
    for i in range(env.num_layers):
        a_k, b_k = env.layer_bounds[i]
        layer_loss = np.clip(claim_size - a_k, 0, b_k - a_k)
        retained_loss += env.alpha_k[i] * layer_loss
    total_retained_loss += retained_loss
    return total_retained_loss

# Example Usage
if __name__ == "__main__":
    # Plot Surplus Distribution
    plot_surplus_distribution(metrics["surplus_history"])

    # Generate and Plot Claim Size Distribution
    training_claims = np.random.lognormal(mean=3.5, sigma=1.0, size=5000) / 100
    out_of_sample_claims = np.random.lognormal(mean=3.7, sigma=1.1, size=5000) / 100
    plot_claim_size_distribution(training_claims, out_of_sample_claims)

    # Plot Ruin Events
    plot_ruin_events(metrics["surplus_history"])

    # Compute Additional Metrics
    total_retained_losses = np.sum([calculate_retained_loss(env, c) for c in out_of_sample_claims])
    total_claims = np.sum(out_of_sample_claims)
    additional_metrics = compute_metrics(metrics["surplus_history"], total_retained_losses, total_claims)
    print("Additional Metrics:")
    print(f"Variance of Surplus: {additional_metrics['variance_surplus']:.2f}")
    print(f"Claim Coverage Efficiency: {additional_metrics['coverage_efficiency']:.2%}")

    # Sensitivity Analysis
    mu_values = [3.5, 3.6, 3.7]
    sigma_values = [1.0, 1.1, 1.2]
    sensitivity_results = sensitivity_analysis(env, generative_model, mu_values, sigma_values)
    print("Sensitivity Analysis Results:")
    for (mu, sigma), result in sensitivity_results.items():
        print(f"\nμ={mu}, σ={sigma}")
        print(f"Mean Surplus: {result['mean_surplus']:.2f}")
        print(f"Ruin Probability: {result['ruin_probability']:.2%}")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Enhanced Visualization and Additional Metrics Functions
def plot_surplus_distribution(surplus_history):
    """
    Plot a histogram of surplus values to assess variability.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(surplus_history, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Surplus Distribution")
    plt.xlabel("Surplus")
    plt.ylabel("Frequency")
    plt.show()

def plot_claim_size_distribution(training_claims, out_of_sample_claims):
    """
    Plot training and out-of-sample claim size distributions side by side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Training Claims
    axes[0].hist(training_claims, bins=30, alpha=0.7, label="Training Claims", color='blue', edgecolor='k')
    axes[0].set_title("Training Claims Distribution", fontsize=14)
    axes[0].set_xlabel("Claim Size", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)

    # Out-of-Sample Claims
    axes[1].hist(out_of_sample_claims, bins=30, alpha=0.7, label="Out-of-Sample Claims", color='green', edgecolor='k')
    axes[1].set_title("Out-of-Sample Claims Distribution", fontsize=14)
    axes[1].set_xlabel("Claim Size", fontsize=12)

    # Final adjustments
    for ax in axes:
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_ruin_events(surplus_history):
    """
    Plot surplus over time, marking potential ruin points.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(surplus_history, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.title("Ruin Events Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Surplus")
    plt.legend()
    plt.show()

def compute_metrics(surplus_history, total_retained_losses, total_claims):
    """
    Compute additional metrics: variance of surplus and claim coverage efficiency.

    Parameters:
        surplus_history (list): History of surplus values.
        total_retained_losses (float): Sum of all retained losses.
        total_claims (float): Sum of all claims.

    Returns:
        dict: Metrics including variance of surplus and claim coverage efficiency.
    """
    variance_surplus = np.var(surplus_history)
    coverage_efficiency = total_retained_losses / total_claims if total_claims > 0 else 0

    return {
        "variance_surplus": variance_surplus,
        "coverage_efficiency": coverage_efficiency
    }

def sensitivity_analysis(env, generative_model, mu_values, sigma_values, num_steps=200):
    """
    Perform sensitivity analysis by tweaking μ and σ.

    Parameters:
        env (InsurerEnv): The insurance environment.
        generative_model (GenerativeClaimModel): The trained generative model.
        mu_values (list): List of μ values to test.
        sigma_values (list): List of σ values to test.
        num_steps (int): Number of steps to simulate for each configuration.

    Returns:
        dict: Sensitivity results for each configuration.
    """
    results = {}

    for mu in mu_values:
        for sigma in sigma_values:
            metrics = out_of_sample_testing(env, generative_model, mu=mu, sigma=sigma, num_steps=num_steps)
            results[(mu, sigma)] = metrics

    return results

# Fixing issue with numpy.float64 in retained loss calculation
def calculate_retained_loss(env, claim):
    """
    Calculate the retained loss for a single claim.
    """
    total_retained_loss = 0
    claim_size = claim * 100
    retained_loss = 0
    for i in range(env.num_layers):
        a_k, b_k = env.layer_bounds[i]
        layer_loss = np.clip(claim_size - a_k, 0, b_k - a_k)
        retained_loss += env.alpha_k[i] * layer_loss
    total_retained_loss += retained_loss
    return total_retained_loss

# Example Usage
if __name__ == "__main__":
    # Plot Surplus Distribution
    plot_surplus_distribution(metrics["surplus_history"])

    # Generate and Plot Claim Size Distribution
    training_claims = np.random.lognormal(mean=3.5, sigma=1.0, size=5000) / 100
    out_of_sample_claims = np.random.lognormal(mean=3.7, sigma=1.1, size=5000) / 100
    plot_claim_size_distribution(training_claims, out_of_sample_claims)

    # Plot Ruin Events
    plot_ruin_events(metrics["surplus_history"])

    # Compute Additional Metrics
    total_retained_losses = np.sum([calculate_retained_loss(env, c) for c in out_of_sample_claims])
    total_claims = np.sum(out_of_sample_claims)
    additional_metrics = compute_metrics(metrics["surplus_history"], total_retained_losses, total_claims)
    print("Additional Metrics:")
    print(f"Variance of Surplus: {additional_metrics['variance_surplus']:.2f}")
    print(f"Claim Coverage Efficiency: {additional_metrics['coverage_efficiency']:.2%}")

    # Sensitivity Analysis
    mu_values = [3.5, 3.6, 3.7]
    sigma_values = [1.0, 1.1, 1.2]
    sensitivity_results = sensitivity_analysis(env, generative_model, mu_values, sigma_values)
    print("Sensitivity Analysis Results:")
    for (mu, sigma), result in sensitivity_results.items():
        print(f"\nμ={mu}, σ={sigma}")
        print(f"Mean Surplus: {result['mean_surplus']:.2f}")
        print(f"Ruin Probability: {result['ruin_probability']:.2%}")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Enhanced Visualization and Additional Metrics Functions
def plot_surplus_distribution(surplus_history):
    """
    Plot a histogram of surplus values to assess variability.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(surplus_history, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Surplus Distribution", fontsize=18)
    plt.xlabel("Surplus", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig("surplus_distribution.png")
    plt.show()

def plot_claim_size_distribution(training_claims, out_of_sample_claims):
    """
    Plot training and out-of-sample claim size distributions side by side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # Training Claims
    axes[0].hist(training_claims, bins=30, alpha=0.7, label="Training Claims", color='blue', edgecolor='k')
    axes[0].set_title("Training Claims Distribution", fontsize=18)
    axes[0].set_xlabel("Claim Size", fontsize=16)
    axes[0].set_ylabel("Frequency", fontsize=16)
    axes[0].tick_params(axis='both', labelsize=14)

    # Out-of-Sample Claims
    axes[1].hist(out_of_sample_claims, bins=30, alpha=0.7, label="Out-of-Sample Claims", color='green', edgecolor='k')
    axes[1].set_title("Out-of-Sample Claims Distribution", fontsize=18)
    axes[1].set_xlabel("Claim Size", fontsize=16)
    axes[1].tick_params(axis='both', labelsize=14)

    # Final adjustments
    for ax in axes:
        ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("claim_size_distribution.png")
    plt.show()

def plot_ruin_events(surplus_history):
    """
    Plot surplus over time, marking potential ruin points.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(surplus_history, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.title("Ruin Events Over Time", fontsize=18)
    plt.xlabel("Steps", fontsize=16)
    plt.ylabel("Surplus", fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig("ruin_events.png")
    plt.show()

def compute_metrics(surplus_history, total_retained_losses, total_claims):
    """
    Compute additional metrics: variance of surplus and claim coverage efficiency.

    Parameters:
        surplus_history (list): History of surplus values.
        total_retained_losses (float): Sum of all retained losses.
        total_claims (float): Sum of all claims.

    Returns:
        dict: Metrics including variance of surplus and claim coverage efficiency.
    """
    variance_surplus = np.var(surplus_history)
    coverage_efficiency = total_retained_losses / total_claims if total_claims > 0 else 0

    return {
        "variance_surplus": variance_surplus,
        "coverage_efficiency": coverage_efficiency
    }

def sensitivity_analysis(env, generative_model, mu_values, sigma_values, num_steps=200):
    """
    Perform sensitivity analysis by tweaking μ and σ.

    Parameters:
        env (InsurerEnv): The insurance environment.
        generative_model (GenerativeClaimModel): The trained generative model.
        mu_values (list): List of μ values to test.
        sigma_values (list): List of σ values to test.
        num_steps (int): Number of steps to simulate for each configuration.

    Returns:
        dict: Sensitivity results for each configuration.
    """
    results = {}

    for mu in mu_values:
        for sigma in sigma_values:
            metrics = out_of_sample_testing(env, generative_model, mu=mu, sigma=sigma, num_steps=num_steps)
            results[(mu, sigma)] = metrics

    return results

# Fixing issue with numpy.float64 in retained loss calculation
def calculate_retained_loss(env, claim):
    """
    Calculate the retained loss for a single claim.
    """
    total_retained_loss = 0
    claim_size = claim * 100
    retained_loss = 0
    for i in range(env.num_layers):
        a_k, b_k = env.layer_bounds[i]
        layer_loss = np.clip(claim_size - a_k, 0, b_k - a_k)
        retained_loss += env.alpha_k[i] * layer_loss
    total_retained_loss += retained_loss
    return total_retained_loss

# Example Usage
if __name__ == "__main__":
    # Plot Surplus Distribution
    plot_surplus_distribution(metrics["surplus_history"])

    # Generate and Plot Claim Size Distribution
    training_claims = np.random.lognormal(mean=3.5, sigma=1.0, size=5000) / 100
    out_of_sample_claims = np.random.lognormal(mean=3.7, sigma=1.1, size=5000) / 100
    plot_claim_size_distribution(training_claims, out_of_sample_claims)

    # Plot Ruin Events
    plot_ruin_events(metrics["surplus_history"])

    # Compute Additional Metrics
    total_retained_losses = np.sum([calculate_retained_loss(env, c) for c in out_of_sample_claims])
    total_claims = np.sum(out_of_sample_claims)
    additional_metrics = compute_metrics(metrics["surplus_history"], total_retained_losses, total_claims)
    print("Additional Metrics:")
    print(f"Variance of Surplus: {additional_metrics['variance_surplus']:.2f}")
    print(f"Claim Coverage Efficiency: {additional_metrics['coverage_efficiency']:.2%}")

    # Sensitivity Analysis
    mu_values = [3.5, 3.6, 3.7]
    sigma_values = [1.0, 1.1, 1.2]
    sensitivity_results = sensitivity_analysis(env, generative_model, mu_values, sigma_values)
    print("Sensitivity Analysis Results:")
    for (mu, sigma), result in sensitivity_results.items():
        print(f"\nμ={mu}, σ={sigma}")
        print(f"Mean Surplus: {result['mean_surplus']:.2f}")
        print(f"Ruin Probability: {result['ruin_probability']:.2%}")


# COMMAND ----------

