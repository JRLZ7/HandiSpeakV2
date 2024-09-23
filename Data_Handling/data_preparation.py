import cv2
import mediapipe as mp
import os
import numpy as np

def collect_data(data_dir, num_classes=29, images_per_class=250):
    """
    Collects images for each class using the computer's camera.
    Recording starts when the user presses 'q'.
    The user can quit the program by pressing 'e'.
    Saves images to the specified directory with hand landmarks drawn, resized to 400x400.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

    for class_index in range(num_classes):
        class_path = os.path.join(data_dir, str(class_index))
        os.makedirs(class_path, exist_ok=True)
        print(f"Set up for class {class_index}. Press 'q' to start collecting, 'e' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                continue
            
            cv2.putText(frame, "Press 'q' to start, 'e' to exit.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                print(f"Starting collection for class {class_index}...")
                break
            elif key == ord('e'):
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                return

        for img_index in range(images_per_class):
            ret, image = cap.read()
            if not ret:
                print("Failed to capture image.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            annotated_image = np.zeros(image.shape, dtype=np.uint8)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
                    )

            annotated_image_resized = cv2.resize(annotated_image, (400, 400))
            save_path = os.path.join(class_path, f'{img_index}.jpg')
            cv2.imwrite(save_path, annotated_image_resized)
            print(f'Saved {save_path}')
            cv2.imshow('Recording', annotated_image_resized)
            if cv2.waitKey(1) == ord('e'):
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATA_DIR = './data/ASL'
    collect_data(DATA_DIR)