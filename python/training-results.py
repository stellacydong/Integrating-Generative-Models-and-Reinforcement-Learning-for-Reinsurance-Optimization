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

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Directory to save results and plots
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

class MetricsCollector:
    def __init__(self):
        self.surplus_log = []
        self.retained_loss_log = []
        self.claims_log = []
        self.alpha_log = []
        self.layer_bounds_log = []
        self.reward_log = []
        self.episode_metrics = []

    def collect_step_metrics(self, surplus, retained_loss, claims, alpha_k, layer_bounds, reward):
        self.surplus_log.append(surplus)
        self.retained_loss_log.append(retained_loss)
        self.claims_log.append(claims)
        self.alpha_log.append(alpha_k.copy())
        self.layer_bounds_log.append(layer_bounds.copy())
        self.reward_log.append(reward)

    def summarize_episode(self):
        episode_summary = {
            "final_surplus": self.surplus_log[-1],
            "total_retained_loss": sum(self.retained_loss_log),
            "total_claims": sum(self.claims_log),
            "average_reward": np.mean(self.reward_log),
        }
        self.episode_metrics.append(episode_summary)
        self.reset_step_metrics()

    def reset_step_metrics(self):
        self.surplus_log = []
        self.retained_loss_log = []
        self.claims_log = []
        self.alpha_log = []
        self.layer_bounds_log = []
        self.reward_log = []

    def save_results(self):
        # Save step-wise logs
        pd.DataFrame({
            "surplus": self.surplus_log,
            "retained_loss": self.retained_loss_log,
            "claims": self.claims_log,
            "reward": self.reward_log
        }).to_csv(os.path.join(output_dir, "step_metrics.csv"), index=False)

        # Save episode metrics
        pd.DataFrame(self.episode_metrics).to_csv(os.path.join(output_dir, "episode_metrics.csv"), index=False)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.surplus_log, label="Surplus Over Time", linewidth=2)
        plt.axhline(y=-100, color="red", linestyle="--", label="Ruin Threshold")
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel("Surplus", fontsize=14)
        plt.title("Surplus Progression", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "surplus_plot.png"))

        plt.figure(figsize=(10, 6))
        plt.hist(self.claims_log, bins=30, alpha=0.7, color='blue')
        plt.xlabel("Claim Sizes", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Claims", fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "claim_distribution.png"))

        plt.figure(figsize=(10, 6))
        retained_loss_per_step = pd.Series(self.retained_loss_log).rolling(window=10).mean()
        plt.plot(retained_loss_per_step, label="Smoothed Retained Loss", linewidth=2)
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel("Retained Loss", fontsize=14)
        plt.title("Retained Loss Over Time", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "retained_loss_plot.png"))

if __name__ == "__main__":
    metrics_collector = MetricsCollector()

    # Simulated example of metric collection during a single episode
    for step in range(200):
        # Example values for demonstration
        surplus = 20000 - step * 100  # Dummy surplus
        retained_loss = np.random.uniform(0, 500)  # Dummy retained loss
        claims = np.random.poisson(10)  # Dummy claims count
        alpha_k = [0.3 + 0.01 * step for _ in range(5)]  # Dummy alpha values
        layer_bounds = [(0 + step, 50 + step) for _ in range(5)]  # Dummy layer bounds
        reward = surplus / 1000 - retained_loss / 100  # Dummy reward

        metrics_collector.collect_step_metrics(surplus, retained_loss, claims, alpha_k, layer_bounds, reward)

    # Summarize at the end of an episode
    metrics_collector.summarize_episode()

    # Save and plot results
    metrics_collector.save_results()
    metrics_collector.plot_results()

    print(f"Results and plots saved to: {output_dir}")




