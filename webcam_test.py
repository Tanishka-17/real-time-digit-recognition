import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mnist_model.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam for mirror effect
    frame = cv2.flip(frame, 1)

    # Prediction area
    x1, y1 = 200, 100
    x2, y2 = 400, 300

    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Show digit here",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Digit Recognizer", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()