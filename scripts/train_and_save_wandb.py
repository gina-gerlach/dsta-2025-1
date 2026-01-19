"""
Script to train and save the MNIST model with Weights & Biases logging (Milestone 4 - Task 2).

This script can be run locally or in a Docker container.
It will:
- Login to W&B using environment variable WANDB_TOKEN
- Train multiple model variants (experiments)
- Save and upload the trained models
- Log metrics and loss to W&B
"""

import os
import wandb
from src.train_model import train_model
from src.model_io import save_model
from tensorflow.keras.callbacks import Callback
import subprocess


class WandbModelCheckpoint(Callback):
    """Custom callback to log metrics to W&B after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log(logs)


def get_git_commit_hash():
    """Retrieve current Git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def main():
    print("=" * 60)
    print("Training MNIST Model with W&B Logging (Task 2)")
    print("=" * 60)

    # Login to W&B using token from environment variable
    wandb_token = os.getenv("WANDB_TOKEN")
    if wandb_token is None:
        raise EnvironmentError("WANDB_TOKEN not found in environment variables.")
    wandb.login(key=wandb_token)

    # Define experiment configurations
    experiments = [
        {"epochs": 5, "batch_size": 128, "optimizer": "adam", "extra_layer": False},
        {"epochs": 5, "batch_size": 128, "optimizer": "sgd", "extra_layer": False},
        {"epochs": 5, "batch_size": 64, "optimizer": "adam", "extra_layer": True},
        {"epochs": 5, "batch_size": 64, "optimizer": "sgd", "extra_layer": True},
    ]

    # Create model directory
    model_dir = os.getenv("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Loop over each experiment
    for config in experiments:
        print(f"\nStarting new W&B run with config: {config}")
        run = wandb.init(
            project="mnist_task2",
            config=config
        )
        run.config.git_commit = get_git_commit_hash()

        # Train the model (ensure train_model accepts optimizer and extra_layer arguments)
        model = train_model(
            epochs=run.config.epochs,
            batch_size=run.config.batch_size,
            optimizer=run.config.optimizer,
            extra_layer=run.config.extra_layer,
            callbacks=[WandbModelCheckpoint()]
        )

        # Save the model
        model_name = f"mnist_model_{run.id}.keras"
        model_path = os.path.join(model_dir, model_name)
        print(f"\nSaving model to {model_path}...")
        save_model(model, filepath=model_path)

        # Upload model artifact to W&B
        artifact = wandb.Artifact(f"mnist_model_{run.id}", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)

        print(f"Run {run.id} complete. Model saved and logged.\n")
        run.finish()

    print("\n" + "=" * 60)
    print("All experiments completed and logged to W&B!")
    print("=" * 60)


if __name__ == "__main__":
    main()