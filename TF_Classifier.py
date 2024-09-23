import time
import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

interpreter = tf.lite.Interpreter(model_path="saved_model/model_quantized.tflite", num_threads=4)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

knuckle_color = (255, 0, 0)  # Blue for knuckles
connection_color = (0, 255, 0)  # Green for connections

label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del',
    5: 'E', 6: 'enter', 7: 'F', 8: 'G', 9: 'H',
    10: 'I', 11: 'J', 12: 'K', 13: 'L', 14: 'M',
    15: 'N', 16: 'O', 17: 'P', 18: 'Q', 19: 'R',
    20: 'rep', 21: 'S', 22: 'space', 23: 'T', 24: 'U',
    25: 'V', 26: 'W', 27: 'X', 28: 'Y', 29: 'Z'
}

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Initialize all timing variables at the start of the loop
    # preprocessing_time = 0
    # drawing_time = 0
    # model_preprocessing_time = 0
    # inference_time = 0
    # postprocessing_time = 0

    # preprocessing_start_time = time.time()
    img = cv.flip(img, 1)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # preprocessing_time = time.time() - preprocessing_start_time

    if results.multi_hand_landmarks:
        # drawing_start_time = time.time()
        frame = np.zeros_like(img)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=knuckle_color, thickness=2, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1))
        # drawing_time = time.time() - drawing_start_time

        # model_preprocessing_start_time = time.time()
        frame_resized = cv.resize(frame, (400, 400)) / 255.0
        frame_input = np.expand_dims(frame_resized, axis=0).astype(np.float32)
        # model_preprocessing_time = time.time() - model_preprocessing_start_time

        # inference_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], frame_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        predicted_label = label_map.get(prediction, "Unknown")
        # inference_time = time.time() - inference_start_time

        # postprocessing_start_time = time.time()
        cv.putText(img, f'Prediction: {predicted_label}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # postprocessing_time = time.time() - postprocessing_start_time
    else:
        cv.putText(img, 'No Hand Detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Hand Gesture Recognition', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Print out the timing information
    # print(f"Preprocessing Time: {preprocessing_time}s, Drawing Time: {drawing_time}s, Model Preprocessing Time: {model_preprocessing_time}s, Inference Time: {inference_time}s, Postprocessing Time: {postprocessing_time}s")

cap.release()
cv.destroyAllWindows()