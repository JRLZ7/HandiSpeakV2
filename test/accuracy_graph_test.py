import matplotlib.pyplot as plt
import numpy as np

# Data
vocab_sizes = [50, 100, 150, 200]
lstm_acc = [94.0, 93.6, 92.53, 86.2]
tsf_acc = [93.6, 92.1, 91.0, 88.15]

# Fit polynomial (degree 1 = linear)
lstm_fit = np.poly1d(np.polyfit(vocab_sizes, lstm_acc, 1))
tsf_fit = np.poly1d(np.polyfit(vocab_sizes, tsf_acc, 1))

# Smooth x for trend lines
x_smooth = np.linspace(min(vocab_sizes), max(vocab_sizes), 200)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(vocab_sizes, lstm_acc, color='blue', marker='o')
plt.scatter(vocab_sizes, tsf_acc, color='orange', marker='o')
plt.plot(x_smooth, lstm_fit(x_smooth), color='blue', linestyle='-', label="LSTM (y=97.7-0.0489x, R^2=0.75)")
plt.plot(x_smooth, tsf_fit(x_smooth), color='orange', linestyle='-', label="Transformer (y=95.6-0.0349x, R^2=0.96)")

# Labels & Title
plt.xlabel("Vocabulary Size (words)")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs. Vocabulary Size")
plt.xticks(vocab_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save & Show
plt.savefig("accuracy_lstm_vs_tsf.png", dpi=300, bbox_inches="tight")
plt.show()
