Real-Time Gender Detection
This project provides a simple yet effective solution for real-time gender detection using a webcam. The core of the system is a deep learning model built with TensorFlow and Keras, which is trained on a dataset of male and female faces.

📁 Project Files
train.py: This script is responsible for training the convolutional neural network (CNN) model. It handles data loading, preprocessing, model architecture definition, and training. It also includes code to save the trained model and plot the training and validation accuracy and loss.

detect_gender_webcam.py: This script uses the trained model to perform real-time gender detection. It captures video from your webcam, detects faces in each frame, and then predicts the gender of each detected face. The predicted label and confidence score are displayed on the screen.

🚀 How to Run the Project
Prerequisites
You'll need to have the following libraries installed. You can install them using pip:

pip install tensorflow keras opencv-python numpy scikit-learn matplotlib

Additionally, this project uses the cvlib library for face detection. You can install it with:

pip install cvlib

Steps
Prepare your dataset:

Create a directory named gender_dataset_face.

Inside this directory, create two sub-directories: man and woman.

Place your training images of men and women in their respective directories. The train.py script is configured to look for this specific folder structure.

Train the model:

Run the train.py script from your terminal:

python train.py

This script will train the model, save the gender_detection.model file, and generate a plot.png image showing the training progress.

Run the webcam detection:

After the model has been trained and saved, you can run the real-time detection script:

python detect_gender_webcam.py

A window will open displaying your webcam feed with a bounding box and a gender label for each detected face.

Press q to quit the application.

🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) built with Keras. It consists of multiple convolutional layers, followed by batch normalization, activation, and max-pooling layers. The final layers are fully connected to classify the gender. This architecture is designed to efficiently learn and extract features from facial images.

🤝 Contributions
Feel free to contribute to this project by improving the model, adding new features, or fixing bugs.
