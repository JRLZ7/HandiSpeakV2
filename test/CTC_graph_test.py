import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = list(range(1, 11))
val_wer = [97.72, 66.83, 52.55, 43.45, 41.83, 33.15, 31.48, 27.45, 26.32, 19.68]

# Fit polynomial (degree 2 works well for curve)
wer_fit = np.poly1d(np.polyfit(epochs, val_wer, 2))

# Smooth x for trend line
x_smooth = np.linspace(min(epochs), max(epochs), 200)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(epochs, val_wer, color='blue', marker='o')
plt.plot(x_smooth, wer_fit(x_smooth), color='blue', linestyle='-', label="Validation WER (trend)")

# Labels & Title
plt.xlabel("Epoch")
plt.ylabel("Validation WER (%)")
plt.title("Validation WER Over Epochs")
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save & Show
plt.savefig("val_wer_over_epochs.png", dpi=300, bbox_inches="tight")
plt.show()
