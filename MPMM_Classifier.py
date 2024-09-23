import numpy as np
import mediapipe as mp
import cv2 as cv
import time

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
frame = None

output_string = ""

def onResult(result: mp.tasks.vision.GestureRecognizerResult, img: mp.Image, timestamp_ms: int):
    global output_string  # Declare as global to modify the variable defined outside this function
    if len(result.gestures) > 0 and len(result.gestures[0]) > 0:
        gesture_name = result.gestures[0][0].category_name
        print(gesture_name)
        # Append to the string if the gesture is not "del"
        if gesture_name == "del":
            output_string = output_string[:-1]
        else:
            output_string += gesture_name

options = mp.tasks.vision.GestureRecognizerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path='saved_model/gesture_recognizer.task'),
    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_hands = 1,
    min_hand_detection_confidence = 0.3,
    min_hand_presence_confidence = 0.3,
    min_tracking_confidence = 0.3,
    result_callback = onResult
    )

recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# recognize a frame
def recognize_async(frame):
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    recognizer.recognize_async(image = mp_image, timestamp_ms = int(time.time()*1000))

last_time = 0

while cap.isOpened():
    current_time = time.time()
    if current_time - last_time >= 0.75:  # Check if 0.75 seconds have passed
        last_time = current_time
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        if frame is None:
            print("error read")
            break

        recognize_async(frame)

    cv.imshow('Win', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
recognizer.close()

print("Accumulated string:", output_string)