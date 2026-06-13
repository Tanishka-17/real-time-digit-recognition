import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load trained model
model = tf.keras.models.load_model("mnist_model.keras")

# Create window
root = tk.Tk()
root.title("Digit Recognizer")

# Canvas
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

# PIL image (same size as canvas!)
image = Image.new("L", (280, 280), color=255)
draw_image = ImageDraw.Draw(image)


def draw(event):
    x = event.x
    y = event.y

    # Draw on tkinter canvas
    canvas.create_oval(
        x - 8, y - 8,
        x + 8, y + 8,
        fill="black",
        outline="black"
    )

    # Draw on PIL image
    draw_image.ellipse(
        [x - 8, y - 8, x + 8, y + 8],
        fill=0
    )


def predict():
    # Copy original image
    img = image.copy()

    img = ImageOps.invert(img)

    
    bbox = img.getbbox()

    if bbox is None:
        result_label.config(text="Draw something first!")
        return

    img = img.crop(bbox)

   
    img = ImageOps.pad(
        img,
        (20, 20),
        color=0
    )

    
    img = img.resize((28, 28))

    img.show()

    
    img_array = np.array(img).astype("float32")

  
    img_array /= 255.0

   
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    # Top 3 guesses
    top3 = np.argsort(prediction[0])[-3:][::-1]

    text = ""

    for i in top3:
        text += f"{i}: {prediction[0][i] * 100:.2f}%\n"

    result_label.config(text=text)


def clear():
    global image, draw_image

    # Clear tkinter canvas
    canvas.delete("all")

    # Reset PIL image
    image = Image.new("L", (280, 280), color=255)
    draw_image = ImageDraw.Draw(image)

    result_label.config(text="Prediction: ?")


canvas.bind("<B1-Motion>", draw)

result_label = tk.Label(
    root,
    text="Prediction: ?",
    font=("Arial", 12)
)
result_label.pack()

predict_button = tk.Button(
    root,
    text="Predict",
    command=predict
)
predict_button.pack()

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear
)
clear_button.pack()

root.mainloop()