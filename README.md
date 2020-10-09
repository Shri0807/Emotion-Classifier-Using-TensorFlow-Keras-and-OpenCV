# Emotion Detection Using TensorFlow, Keras and OpenCV

This project is a simple Emotion Detection used to detect emotions of 5 classes on a Human Face. The Convolutional Neural Network was built using TensorFlow, Keras and OpenCV Python.

Main Libraries used for Project:-

* TensorFlow 2.0.0
* Keras 2.3.1
* OpenCV 4.2.*

Other Libraries include Numpy, scikit-learn, Matplotlib.

The Project has 2 files:
1. train.py
2. test.py

-----------------------------------------------------------------

## Dataset
- Kaggle Dataset :- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.

------------------------------------------------------------------

## Train.py

This python file is used for training the Deep learning Neural Network. The images are loaded into the file using Keras flow_from_directory() function. Each Image is preprocessed and converted to numpy array. The imgaes are normalized for better accuracy. ImageDataGenerator() function is used for Data Augmentation.

Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It helps reduce overfitting when training a machine learning. It is clesely related to oversampling in data analysis.

Techniques such as Earlystopping is also used to improve the accuracy of the model and also assures that the model does not overfit the data. The Model is built over MobileNet. The Layers of MobileNet are made trainable so that we can train the model. 

Note:
> The only Prerequisite for MobileNet is that the image needs to of size 224x224. Remember to resize the image when using MobileNet.

The model saved as Emotion_Detection.h5. The trained model is later loaded into test.py for testing.

------------------------------------------------------------------

## Test.py
This file uses OpenCV and Numpy. The Trained Model is loaded using Keras load_model() function. Using cv2.VideoCapture() function the video from webcam is captured. Five class labels are created to classify Emotions namely: Angry, Happy, Neutral, Sad, Surprise. 

haarcascade_frontalface_default.xml is used to detect the face of a person. The file is available in the Official Github Repository of OpenCV.

The Output Folder has Outputs of all 5 types of emotion.

-------------------------------------------------------------------
# How to Run the Project
1. Create a python virtual environment and install the dependencies.
2. Run the train.py file for training the Model.
3. Once training is completed Run test.py for testing the Model.