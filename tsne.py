import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the latent embeddings your VAE generated
train_z = np.load('ctabsyn/tabsyn/vae/ckpt/adult/train_z.npy')

# Flatten it if it's 3D: [batch, features, token_dim] -> [batch, latent_dim]
if len(train_z.shape) == 3:
    train_z = train_z.reshape(train_z.shape[0], -1)

# Load the real labels
y_train = np.load('data/adult/Tabsyn/y_train.npy').reshape(-1)

print("Running t-SNE (This may take 1-2 minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(train_z)

plt.figure(figsize=(10, 8))

# 1. Plot Majority (0)
plt.scatter(z_2d[y_train==0, 0], z_2d[y_train==0, 1], alpha=0.3, label='Majority (0)', color='blue', s=5)

# 2. Plot Overlap (1) - Added this missing class!
plt.scatter(z_2d[y_train==1, 0], z_2d[y_train==1, 1], alpha=0.5, label='Overlap (1)', color='green', s=10)

# 3. Plot Minority (2) - Updated label and condition
plt.scatter(z_2d[y_train==2, 0], z_2d[y_train==2, 1], alpha=0.8, label='Minority (2)', color='red', s=15)

plt.title("t-SNE of VAE Latent Space (Z)")
plt.legend()
plt.savefig('ctabsyn_tsne.png', dpi=300, bbox_inches='tight')
print("Plot successfully saved to cfg_tsne.png!")