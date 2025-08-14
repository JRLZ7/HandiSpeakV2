import matplotlib.pyplot as plt

# Data
vocab_sizes = [50, 100, 150, 200]
lstm_acc = [94.0, 93.6, 92.53, 86.2]
tsf_acc = [93.6, 92.1, 91.0, 88.15]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(vocab_sizes, lstm_acc, marker='o', linestyle='-', label="LSTM", color='blue')
plt.plot(vocab_sizes, tsf_acc, marker='o', linestyle='-', label="Transformer", color='orange')

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
