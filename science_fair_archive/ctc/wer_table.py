import pandas as pd
import numpy as np

# Define your references, predictions, and WER calculation
references = [
    [8, 13, 27, 46, 23, 8, 24],
    [40, 2, 24, 26, 17, 13, 33],
    [32, 29, 35, 20, 9, 18, 27],
    [45, 7, 30, 30, 20, 2, 1],
    [40, 41, 32, 11, 37, 40]
]

predictions = [
    [30, 45, 30, 45, 30, 10, 45, 30, 4, 45, 4, 0, 30, 4, 30, 45, 30, 4, 12, 30, 45, 4, 12, 30, 45, 30, 4, 0, 4, 20, 4, 30],
    [10, 22, 10, 22, 10, 22, 10, 22, 34, 22, 30, 22, 10, 30, 10, 22, 10, 30, 10],
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 22, 22, 22, 10, 10, 10, 10, 10, 10],
    [10, 22, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 7, 7, 7, 7, 7, 7],
    [24, 17, 24, 17, 24, 22, 46, 22, 46, 22, 24, 17, 24, 10, 24, 10, 24]
]

# Function to calculate WER as percentage
def calculate_wer(ref, pred):
    ref_len = len(ref)
    pred_len = len(pred)
    dp = np.zeros((ref_len + 1, pred_len + 1))

    for i in range(ref_len + 1):
        for j in range(pred_len + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Return WER as a percentage
    return (dp[ref_len][pred_len] / ref_len) * 100

# Calculate WER for each reference-prediction pair
wer_list = [calculate_wer(ref, pred) for ref, pred in zip(references, predictions)]

# Create a pandas DataFrame for the table
wer_df = pd.DataFrame({
    'Reference': [' '.join(map(str, ref)) for ref in references],
    'Prediction': [' '.join(map(str, pred)) for pred in predictions],
    'WER (%)': [round(wer, 3) for wer in wer_list]
})

# Display the WER table
print(wer_df)

# Example WER values in percentage form
WER_values = [4.571, 2.714, 4.714, 2.571, 2.833]
average_WER = sum(WER_values) / len(WER_values)
print(f"Average WER: {round(average_WER, 3)} %")
