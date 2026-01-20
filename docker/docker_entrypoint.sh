#!/bin/bash
# Exit immediately if a command fails
set -e

# Load .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Login to W&B using token
if [ -z "$WANDB_TOKEN" ]; then
  echo "Error: WANDB_TOKEN not found. Please set it in the .env file."
  exit 1
fi

echo "Logging into Weights & Biases..."
wandb login $WANDB_TOKEN

# Execute the command passed to the container
exec "$@"
