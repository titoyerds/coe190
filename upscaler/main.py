import mlstac
import torch
import cubo
import matplotlib.pyplot as plt
import numpy as np

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
  output_dir="model/SEN2SRLite_RGBN",
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)
model = model.to(device)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=8.142535,
    lon=122.859598,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B08"],
    start_date="2025-10-20",
    end_date="2025-10-20",
    edge_size=128,
    resolution=10
)

# Prepare the data to be used in the model, select just one sample 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Apply model
superX = model(X[None]).squeeze(0)


# Convert tensors back to NumPy
superX_np = superX.detach().cpu().numpy()
orig_np = X.detach().cpu().numpy()

# Normalize to 0–1 for display
superX_np = np.clip(superX_np, 0, 1)
orig_np = np.clip(orig_np, 0, 1)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.transpose(orig_np[:3], (1, 2, 0)))  # RGB from Sentinel-2
plt.title("S2 Before - 10m")

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(superX_np[:3], (1, 2, 0)))  # RGB from SEN2SRLite output
plt.title("SEN2SRLite After - 2.5m")

plt.tight_layout()
plt.show()
