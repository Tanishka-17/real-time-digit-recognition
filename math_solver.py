import matplotlib.pyplot as plt
import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps


model = tf.keras.models.load_model("mnist_model.keras")


root = tk.Tk()
root.title("Digit Recognizer")


canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()


image = Image.new("L", (280, 280), color=255)
draw_image = ImageDraw.Draw(image)


def draw(event):
    x = event.x
    y = event.y

    
    canvas.create_oval(
        x - 8, y - 8,
        x + 8, y + 8,
        fill="black",
        outline="black"
    )

    
    draw_image.ellipse(
        [x - 8, y - 8, x + 8, y + 8],
        fill=0
    )


def predict():
    
    img = image.copy()

    img = ImageOps.invert(img)

    
    bbox = img.getbbox()

    if bbox is None:
        result_label.config(text="Draw something first!")
        return

    img = img.crop(bbox)

   
    img.thumbnail((20, 20))
    new_img = Image.new("L", (28, 28), 0)
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2
    new_img.paste(img, (x, y))
    img = new_img
    
    plt.imshow(img, cmap="gray")
    plt.title("What the AI sees")
    plt.axis("off")
    plt.show()

    img.show()

    
    img_array = np.array(img).astype("float32")

  
    img_array /= 255.0

   
    img_array = img_array.reshape(1, 28, 28)

    
    prediction = model.predict(img_array, verbose=0)

    
    top3 = np.argsort(prediction[0])[-3:][::-1]

    text = ""

    for i in top3:
        text += f"{i}: {prediction[0][i] * 100:.2f}%\n"

    result_label.config(text=text)


def clear():
    global image, draw_image

    
    canvas.delete("all")

   
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