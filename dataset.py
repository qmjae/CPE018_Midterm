import cv2
import os

# Create a directory to save the images
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
image_count = 0
image_size = (256, 256)

while True:
    ret, frame = cap.read()

    if not ret:
        print("CAMERA IS FAILED CANT CAPTURE")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], image_size)

        if cv2.waitKey(1) & 0xFF == ord('p'):  # Change to 'p' for capturing images
            image_count += 1
            image_name = os.path.join(output_dir, f"EJERCITO{image_count}.png")  # Change the extension to .png
            cv2.imwrite(image_name, face)  # Save as PNG format

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Change to 'q' for exiting the loop
        break

cap.release()
cv2.destroyAllWindows()
