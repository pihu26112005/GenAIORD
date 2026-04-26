import os

print("Starting TabDDPM Training using optimized paper hyperparameters...")

# Pointing to the pre-existing best configuration
config_path = "exp/adult/ddpm_mlp_best/config.toml"

os.system(f"python scripts/pipeline.py --config {config_path} --train")
print("Training Complete!")


# tmux new -s tabddpm_train
# python train_model.py
# # Press Ctrl+B, then D to detach