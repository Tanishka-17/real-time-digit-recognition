# Real-Time Handwritten Digit Recognition

A computer vision project that recognizes handwritten digits in real time using a webcam.

## Features

- CNN trained on the MNIST dataset
- OpenCV webcam integration
- Adaptive thresholding for different lighting conditions
- Contour-based digit extraction
- MNIST-compatible preprocessing
- Live predictions with confidence scores

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## How It Works

1. Capture webcam feed.
2. Extract the digit from the Region of Interest.
3. Convert it into a 28×28 MNIST-style image.
4. Feed it into the CNN.
5. Display prediction and confidence score live.

## Demo

Show a handwritten digit inside the green box and the system predicts it in real time.