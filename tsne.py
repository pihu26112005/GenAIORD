import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the latent embeddings your VAE generated
train_z = np.load('latent-ctabsyn/tabsyn/vae/ckpt/adult/train_z.npy')

# Flatten it if it's 3D: [batch, features, token_dim] -> [batch, latent_dim]
if len(train_z.shape) == 3:
    train_z = train_z.reshape(train_z.shape[0], -1)

# Load the real labels
y_train = np.load('data/adult/y_train.npy').reshape(-1)

print("Running t-SNE (This may take 1-2 minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(train_z)

plt.figure(figsize=(10, 8))
# Plot Majority (0)
plt.scatter(z_2d[y_train==0, 0], z_2d[y_train==0, 1], alpha=0.3, label='Majority (0)', color='blue', s=5)
# Plot Minority (1)
plt.scatter(z_2d[y_train==1, 0], z_2d[y_train==1, 1], alpha=0.8, label='Minority (1)', color='red', s=10)

plt.title("t-SNE of VAE Latent Space (Z)")
plt.legend()
plt.savefig('latent_tsne.png', dpi=300, bbox_inches='tight')
print("Plot successfully saved to latent_tsne.png!")