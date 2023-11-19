import numpy as np
import os
import cv2
import sys
from sklearn.svm import SVC

def read_images(path):
    X, y = [], []
    names = []

    orb = cv2.ORB_create()

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    print(f"Processing file: {filename} in directory: {subject_path}")
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    resized_im = cv2.resize(im, (200, 200))

                    keypoints, descriptors = orb.detectAndCompute(resized_im, None)

                    if descriptors is not None and len(descriptors) > 0:
                        # Truncate or zero-pad descriptors to have a fixed length (e.g., 128)
                        if len(descriptors) < 128:
                            descriptors = np.vstack([descriptors, np.zeros((128 - len(descriptors), 32), dtype=np.float32)])
                        else:
                            descriptors = descriptors[:128]

                        # Flatten and append descriptors
                        flattened_descriptor = descriptors.flatten()
                        X.append(flattened_descriptor)
                        y.append(subdirname)  # Assuming folder names correspond to emotions
                        names.append(subdirname)
                    else:
                        print(f"No descriptors found for image: {filepath}")

                except IOError as e:
                    print(f"I/O Error({e.errno}): {e.strerror}")
                except Exception as ex:
                    print(f"Unexpected error: {ex}")
                    print(f"Error occurred while processing file: {filename}")
                    raise

    return [X, y, names]

def train_svm(X, y):
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    return svm
def predict_emotion(model, image_data):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image_data, None)

    # Ensure consistent descriptor length
    if descriptors is not None and len(descriptors) < 128:
        descriptors = np.vstack([descriptors, np.zeros((128 - len(descriptors), 32), dtype=np.float32)])
    elif descriptors is not None:
        descriptors = descriptors[:128]

    if descriptors is not None:
        flattened_descriptor = descriptors.flatten()
        prediction_index = model.predict([flattened_descriptor])[0]  # Retrieve the prediction index
        return prediction_index
    else:
        print("No descriptors found for the given image data.")
        return None

def face_rec():
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images>")
        sys.exit()

    [X, y, names] = read_images(sys.argv[1])
    model = train_svm(X, y)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                prediction_index = predict_emotion(model, roi)
                print(f"Prediction Index: {prediction_index}")
                
                if prediction_index is not None:
                    if isinstance(prediction_index, str):
                        print("Prediction index is a string")
                        emotion = prediction_index
                else:
                    print(f"Prediction Index (before conversion): {prediction_index}")
                    emotion = names[int(prediction_index)]  # Convert prediction_index to integer before using it
                    print(f"Emotion Index: {emotion}")
                    
                cv2.putText(img, emotion, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print(f"Emotion: {emotion}")
            except Exception as e:
                print(f"Error: {e}")
                continue

        cv2.imshow("camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()