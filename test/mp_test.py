import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect landmarks
        results = holistic.process(image)

        # Convert the image color back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # Draw hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Print coordinates
        if results.left_hand_landmarks:
            print("Left Hand Landmarks:")
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                print(f"  Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        if results.right_hand_landmarks:
            print("Right Hand Landmarks:")
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                print(f"  Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        # if results.pose_landmarks:
        #     print("Pose Landmarks:")
        #     for i, landmark in enumerate(results.pose_landmarks.landmark):
        #         print(f"  Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

        # Display the image
        cv2.imshow('Holistic Model', image)

        # Quit the loop with 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
