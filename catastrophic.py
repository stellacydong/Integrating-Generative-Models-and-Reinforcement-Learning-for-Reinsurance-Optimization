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

import matplotlib.pyplot as plt
from scipy.stats import pareto

# High-Frequency vs. Low-Frequency Claims Simulation
def simulate_frequency_scenarios(env, model, frequencies, time_steps=200):
    results = {}
    for frequency in frequencies:
        env.lambda_ = frequency
        state = env.reset()
        surplus_log = []

        for step in range(time_steps):
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            surplus_log.append(info['surplus'])
            if done:
                break

        results[frequency] = surplus_log

    return results

# Pandemic Impact Simulation
def simulate_pandemic(env, model, pandemic_duration=50, time_steps=200):
    surplus_log = []
    ruin_probability = 0

    state = env.reset()
    for step in range(time_steps):
        if step < pandemic_duration:
            env.lambda_ = 100  # Significantly higher claim frequency
        else:
            env.lambda_ = 10  # Return to normal frequency

        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        surplus_log.append(info['surplus'])

        if info['surplus'] < -100:
            ruin_probability += 1
            break

        if done:
            break

    ruin_probability /= time_steps
    return surplus_log, ruin_probability

# Catastrophic Tail Event Simulation
def simulate_catastrophe(env, model, time_steps=200):
    surplus_log = []
    ruin_probability = 0

    env.lambda_ = 50  # Increased claim frequency for catastrophe
    env.layer_bounds = [(0, 200)] * env.num_layers  # Widen reinsurance layers

    state = env.reset()
    for step in range(time_steps):
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        surplus_log.append(info['surplus'])

        if info['surplus'] < -100:
            ruin_probability += 1
            break

        if done:
            break

    ruin_probability /= time_steps
    return surplus_log, ruin_probability

# Main Program for Stress Testing
if __name__ == "__main__":
    # High-Frequency vs. Low-Frequency Simulation
    frequencies = [5, 10, 50, 100]  # Different claim frequencies
    frequency_results = simulate_frequency_scenarios(env, model, frequencies)

    # Plot High-Frequency vs. Low-Frequency Results
    plt.figure(figsize=(10, 6))
    for freq, log in frequency_results.items():
        plt.plot(log, label=f"Frequency: {freq}")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Surplus", fontsize=14)
    plt.title("High-Frequency vs. Low-Frequency Claims", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("frequency_results.png")
    plt.show()

    # Pandemic Impact Simulation
    pandemic_log, pandemic_ruin_prob = simulate_pandemic(env, model)

    # Plot Pandemic Impact Results
    plt.figure(figsize=(10, 6))
    plt.plot(pandemic_log, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Surplus", fontsize=14)
    plt.title("Pandemic Impact Simulation", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("pandemic_impact.png")
    plt.show()

    print(f"Pandemic Impact - Final Surplus: ${pandemic_log[-1]:.2f}")
    print(f"Pandemic Impact - Ruin Probability: {pandemic_ruin_prob * 100:.2f}%")

    # Catastrophic Tail Event Simulation
    catastrophe_log, catastrophe_ruin_prob = simulate_catastrophe(env, model)

    # Plot Catastrophic Tail Event Results
    plt.figure(figsize=(10, 6))
    plt.plot(catastrophe_log, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Surplus", fontsize=14)
    plt.title("Catastrophic Tail Event Simulation", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("catastrophic_event.png")
    plt.show()

    print(f"Catastrophic Event - Final Surplus: ${catastrophe_log[-1]:.2f}")
    print(f"Catastrophic Event - Ruin Probability: {catastrophe_ruin_prob * 100:.2f}%")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto

# Catastrophe Risk Simulation Function
def simulate_catastrophe(env, model, time_steps=200):
    surplus_log = []
    ruin_probability = 0

    # Adjust the environment for catastrophe simulation
    env.lambda_ = 50  # Increased claim frequency for extreme risk
    env.layer_bounds = [(0, 200)] * env.num_layers  # Widen reinsurance layers

    state = env.reset()
    for step in range(time_steps):
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        surplus_log.append(info['surplus'])

        # Check for ruin
        if info['surplus'] < -100:
            ruin_probability += 1
            break

        if done:
            break

    ruin_probability /= time_steps
    return surplus_log, ruin_probability

# Pareto Distribution for Catastrophic Claims
def generate_catastrophic_claims(shape, size):
    return pareto.rvs(shape, size=size)

# Main Program for Catastrophe Simulation
if __name__ == "__main__":
    # Create a new environment for catastrophe simulation
    catastrophe_scenario = {
        "T": 10,
        "n": 200,
        "lambda_": 50,  # Increased claim frequency
        "alpha_bounds": (0.2, 0.5),
        "num_layers": 5,
        "budget_max": 150_000,
        "ruin_prob_target": 0.01,
        "generative_model": generative_model,
    }
    catastrophe_env = InsurerEnv(**catastrophe_scenario)

    # Generate catastrophic claim data
    catastrophic_data = generate_catastrophic_claims(shape=2.5, size=5000)
    catastrophic_data = catastrophic_data.reshape(-1, 1) / 100  # Scale claims

    # Train the generative model with catastrophic claims
    generative_model.train(catastrophic_data, epochs=5, batch_size=16)

    # Run the catastrophe simulation
    surplus_log, ruin_probability = simulate_catastrophe(catastrophe_env, model, time_steps=200)

    # Plot Catastrophe Simulation Results
    plt.figure(figsize=(10, 6))
    plt.plot(surplus_log, label="Surplus Over Time")
    plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Surplus", fontsize=14)
    plt.title("Catastrophe Risk Simulation", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("catastrophe_risk.png")
    plt.show()

    # Display final results
    final_surplus = surplus_log[-1] if surplus_log else -100
    print(f"Final Surplus: ${final_surplus:.2f}")
    print(f"Ruin Probability: {ruin_probability * 100:.2f}%")


# COMMAND ----------

