# Real_Time_Face_Detection
This project is a machine learning application that focuses on face mask detection using the TensorFlow and OpenCV libraries. The primary goal is to build a model capable of distinguishing between individuals wearing masks and those without masks. Here's a breakdown of the project:

1) Data Preparation: The project starts by importing necessary libraries and loading a dataset containing images of people with and without masks. Images are preprocessed to a fixed size of 128x128 pixels.

2) Model Architecture: Two popular pre-trained convolutional neural network (CNN) architectures, MobileNetV2 and InceptionV3, are used as base models. These models are extended with additional layers, including flattening, dense layers, and dropout for classification. The final output layer uses sigmoid activation for binary classification.

3) Model Training: The dataset is split into training and testing sets, with an emphasis on maintaining class balance (stratified splitting). The model is trained using binary cross-entropy loss and the Adam optimizer. Training is performed for 10 epochs, with model checkpointing to save the best weights.

4) Validation: A separate validation dataset is prepared to evaluate the model's performance. Images from this dataset are loaded, processed, and used to assess the model's accuracy and other evaluation metrics.

5) Face Detection: OpenCV is used to detect faces in an input image. The Haar Cascade Classifier is employed for this task. Detected faces are then processed, resized to the model's input dimensions, and passed through the trained model for mask detection.

6) Visualization: The project concludes with a visual output that displays the input image with bounding boxes around detected faces. Text labels ('Mask' or 'No Mask') are added to each bounding box to indicate whether the person is wearing a mask or not.

This project combines deep learning, image processing, and computer vision techniques to create a face mask detection system. It can be used for real-time monitoring in various scenarios.
