# numbers-classification-with-cnn
MNIST Digit Classification using Convolutional Neural Network (CNN)
This project demonstrates how to build a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9), each of size 28x28 pixels.

Requirements
TensorFlow
NumPy
Matplotlib
You can install the required packages using pip.

Code Explanation
1. Importing Libraries and Loading Data
In the first step, we import the necessary libraries and load the MNIST dataset. The dataset is split into training and testing sets, and the dimensions of each set are printed for verification.

2. Displaying a Sample Image
We display a sample image from the training data to get a visual understanding of what the dataset looks like. The image is shown in binary format (black and white).

3. Preprocessing the Data
The images in the dataset are converted to floating-point numbers and normalized by dividing by 255. This scaling helps in faster convergence during training.

4. Building the CNN Model
A Convolutional Neural Network (CNN) is constructed using TensorFlow and Keras. The model consists of:

Convolutional layers for feature extraction.
MaxPooling layers for downsampling.
Dropout layers for regularization to prevent overfitting.
A Flatten layer to convert 2D matrices to 1D vectors.
Dense layers for classification with the final layer using a softmax activation function to output probability distributions over the digit classes.
5. Compiling the Model
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. Accuracy is used as the evaluation metric.

6. Reshaping Data for the CNN
The training and testing images are reshaped to include a single channel, as required by the CNN input.

7. Training the Model
The model is trained on the training data for a specified number of epochs. During training, 30% of the training data is used for validation.

8. Evaluating the Model
After training, the model's performance is evaluated on the test set, and the test accuracy is printed.

9. Making Predictions
The trained model is used to make predictions on the test set. The predicted class for the first test image is printed along with the actual class.

10. Visualizing Predictions
The first few test images, along with their predicted and true labels, are displayed. Correct predictions are shown in blue, while incorrect predictions are shown in red.

11. Plotting Training Metrics
Finally, the training and validation accuracy and loss are plotted to visualize the training process and detect any signs of overfitting.

Conclusion
This project provides a comprehensive walkthrough of building, training, and evaluating a Convolutional Neural Network for digit classification using the MNIST dataset. The model achieves high accuracy on the test set, demonstrating the effectiveness of CNNs for image classification tasks.
