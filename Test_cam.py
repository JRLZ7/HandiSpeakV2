import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error while trying to open camera. Please check again...')
    exit()

while cap.isOpened():
    # capture each frame of the video
    ret, frame = cap.read()

    if not ret:
        print('Error: Failed to capture frame')
        break

    # resize the frame (optional)
    frame = cv2.resize(frame, (200, 200))

    cv2.imshow('image', frame)

    # press `q` to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()