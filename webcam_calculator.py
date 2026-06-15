import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("mnist_model.keras")

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_for_mnist(digit_img):
    """Resizes digit to fit inside 20x20 while maintaining aspect ratio, then pads to 28x28."""
    h, w = digit_img.shape
    
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))
        
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- NEW UI: TWO BOXES ---
    # Box 1 (Left)
    b1_x1, b1_y1 = 80, 100
    b1_x2, b1_y2 = 280, 300
    
    # Box 2 (Right)
    b2_x1, b2_y1 = 360, 100
    b2_x2, b2_y2 = 560, 300

    # Draw the boxes and the math symbol
    cv2.rectangle(frame, (b1_x1, b1_y1), (b1_x2, b1_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (b2_x1, b2_y1), (b2_x2, b2_y2), (0, 255, 0), 2)
    
    # Draw a big "+" in the middle
    cv2.putText(frame, "+", (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(frame, "Num 1", (b1_x1, b1_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Num 2", (b2_x1, b2_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def extract_number_from_roi(x1, y1, x2, y2, display_offset_x):
        """Helper function to process a box and return the recognized string of digits."""
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh[:5, :] = 0
        thresh[-5:, :] = 0
        thresh[:, :5] = 0
        thresh[:, -5:] = 0

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        if contours:
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                
                # STRICTER FILTER: Ignores wallpaper patterns and small shadows
                if ch < 30 or area < 150 or cw > 1.5 * ch or ch > (y2 - y1) * 0.9:
                    continue
                    
                digits.append((cx, cy, cw, ch))

            digits = sorted(digits, key=lambda d: d[0])

        result_str = ""
        for cx, cy, cw, ch in digits:
            digit_img = thresh[cy:cy+ch, cx:cx+cw]
            processed_img = preprocess_for_mnist(digit_img)
            
            img_array = processed_img.astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            prediction = model.predict(img_array, verbose=0)
            digit = np.argmax(prediction)
            result_str += str(digit)
            
            # Draw blue boxes on the main frame
            cv2.rectangle(frame, (display_offset_x + cx, y1 + cy), 
                          (display_offset_x + cx + cw, y1 + cy + ch), (255, 0, 0), 2)
            
        return result_str

    # Process both boxes
    num1_str = extract_number_from_roi(b1_x1, b1_y1, b1_x2, b1_y2, b1_x1)
    num2_str = extract_number_from_roi(b2_x1, b2_y1, b2_x2, b2_y2, b2_x1)

    # --- THE CALCULATOR LOGIC ---
    display_text = "Waiting for numbers..."
    
    if num1_str and num2_str:
        try:
            val1 = int(num1_str)
            val2 = int(num2_str)
            total = val1 + val2
            display_text = f"Live Math: {val1} + {val2} = {total}"
        except ValueError:
            pass
    elif num1_str:
        display_text = f"Live Math: {num1_str} + ?"
    elif num2_str:
        display_text = f"Live Math: ? + {num2_str}"

    cv2.putText(frame, display_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Calculator", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()