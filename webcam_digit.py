import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mnist_model.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

  
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

    
    roi = frame[y1:y2, x1:x2]

    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
    )


    contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
    )


    clean = np.zeros_like(thresh)

    if contours:
        clean = np.zeros_like(thresh)

        best = None
        best_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if w > 3 * h:
                continue
            if area < 200:
                continue
            if area > best_area:
                best = cnt
                best_area = area

        if best is not None:
            cv2.drawContours(
                clean,
                [best],
                -1,
                255,
                thickness=cv2.FILLED
        )

        thresh = clean

        

    
    cv2.imshow("AI View", thresh)

    
    img = cv2.resize(thresh, (28, 28))


    img_array = img.astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    cv2.putText(
    frame,
    f"{digit} ({confidence*100:.1f}%)",
    (x1, y2 + 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
   )


    cv2.imshow(
        "28x28",
        cv2.resize(img, (280, 280),
                   interpolation=cv2.INTER_NEAREST)
    )

    cv2.imshow("Digit Recognizer", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()