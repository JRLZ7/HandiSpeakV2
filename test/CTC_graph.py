import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 11))
val_wer = [97.72, 66.83, 52.55, 43.45, 41.83, 33.15, 31.48, 27.45, 26.32, 19.68]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_wer, marker='o', linestyle='-', color='blue', label='Validation WER')

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
