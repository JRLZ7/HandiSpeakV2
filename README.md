# HandiSpeakV2 Pipeline

This repository contains the complete pipeline for processing the WLASL dataset, training multiple sign language recognition models, and running optimized inference.

The steps below outline the exact order in which scripts should be executed to go from raw WLASL videos to a trained and quantized model.

> **Note:** Files inside the `test/` directory are not required for the main pipeline.  
> They were used only for temporary experiments and testing certain functionalities during development.

---

## Script Execution Order

### 1–6: Preprocessing (All in `src/` folder)

1. **`word_finder.py`**  
   Select the top N words (options: `50`, `100`, `150`, `200`).

2. **`data_processing.py`**  
   Generate metadata JSON listing videos for the selected vocabulary.

3. **`data_keypoints.py`**  
   Extract MediaPipe keypoints from each video in the metadata and save per-word JSON files in `keypoints/`.

4. **`data_augment.py`**  
   Apply augmentations (e.g., Gaussian noise) to keypoints and save the expanded dataset in `keypoints_aug/`.

5. **`data_keypoints_aug_sort.py`**  
   Sort augmented keypoints into `50/`, `100/`, `150/`, and `200/` word training folders.

6. **`data_mp_split.py`**  
   Split each vocabulary-size folder into `train/` and `val/` sets.

---

### 7: Model Training

#### LSTM Model (in `LSTM/` folder)
- **`LSTM_data.py`** – Load dataset.  
- **`LSTM_model.py`** – Define LSTM architecture.  
- **`LSTM_train.py`** – Train the LSTM model.  
- **`LSTM_quantize.py`** – Quantize trained LSTM to ONNX INT8 for optimized inference.

#### Transformer Model (TSF) (in `TSF/` folder)
- **`TSF_data.py`** – Load dataset.  
- **`TSF_model.py`** – Define Transformer architecture.  
- **`TSF_train.py`** – Train the Transformer model.  
- **`TSF_quantize.py`** – Quantize trained Transformer to ONNX INT8 for optimized inference.

---

### 8: Inference
- **`src/inference_time.py`**  
  Run inference on a trained and quantized model.

---

## Notes
- The `test/` directory contains experimental scripts for debugging or trying out specific features.  
  These are not part of the main execution pipeline.
- Ensure dataset paths, vocabulary sizes, and parameters are correctly set before running.
- Model training (Step 7) can be repeated for different vocabulary sizes for comparison.
