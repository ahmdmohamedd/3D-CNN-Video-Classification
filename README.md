# 3D CNN Video Classification

This repository provides a comprehensive implementation of a 3D Convolutional Neural Network (CNN) for video classification tasks. The goal of this project is to develop a deep learning model capable of performing action recognition from video data. The system utilizes 3D CNNs to capture both spatial and temporal information in videos for accurate classification of actions across multiple categories.

## Dataset

The primary dataset intended for use in this system is **UCF101**, a widely-used action recognition dataset. It contains **13,320 videos** from 101 action categories, such as "biking," "running," "climbing," and many more. Each video in the dataset has been labeled with one of these actions.

### Dataset Size

- **UCF101 dataset**: Approximately **7GB**.
- The dataset consists of **101 action categories**, with each category containing **about 100 videos**.

### Dataset Limitations

- The system was designed to work with the full UCF101 dataset, but **due to the large size of the dataset (7GB)**, the system has **not been tested with the full dataset**. Users can test the system with subsets of the dataset or any other suitable action recognition dataset.

## Requirements

To run this system, ensure that you have the following software and libraries installed:

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **OpenCV**
- **scikit-learn**
- **NumPy**

You can install the necessary dependencies using `pip`:

```bash
pip install tensorflow keras opencv-python scikit-learn numpy
```

## System Overview

This system leverages the power of 3D Convolutional Neural Networks (CNNs) to classify actions in videos. 3D CNNs extend traditional 2D CNNs by adding a third dimension (time) to the convolution operation, making them particularly well-suited for video data, where both spatial and temporal features need to be captured.

### Architecture

- **Conv3D layers**: The model starts with 3D convolutional layers to capture spatial and temporal features across multiple frames in a video.
- **MaxPooling3D layers**: These layers reduce the dimensionality of the feature maps while retaining the most important features.
- **Dense layers**: The final layers consist of dense layers for classification.
- **Dropout**: To prevent overfitting, a dropout layer is included before the final output layer.

The model is trained on the video frames and learns to classify actions based on patterns observed across time and space in the video clips.

## How to Use

### Step 1: Prepare the Dataset

1. Download the **UCF101 dataset** (or any suitable video action recognition dataset).
   - You can find the UCF101 dataset [here](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) (Note: the dataset is large, ~7GB).
2. Extract the dataset into a local directory on your machine.

### Step 2: Modify Dataset Path

In the script `video_classification.ipynb`, update the `dataset_dir` variable to point to the folder containing the dataset:

```python
dataset_dir = '/path/to/UCF101'  # Update with the path to your dataset
```

### Step 3: Run the Notebook

Open and run the Jupyter notebook `video_classification.ipynb` to:

- Preprocess the videos.
- Build the 3D CNN model.
- Train the model on the dataset.
- Evaluate the model's performance on the test set.

### Step 4: Training

The model training process involves:
- **Video preprocessing**: Videos are read, resized, and converted to frames. A fixed number of frames (16) is sampled from each video.
- **Model training**: The model is trained using the training set, and evaluation is done using the test set.

### Step 5: Model Evaluation

Once trained, the model is evaluated on the test set, and the accuracy is reported. Note that the model's performance may vary depending on the dataset used and the specific configuration.

## Notes

- **Testing**: Due to the large size of the UCF101 dataset (7GB), the system was not tested with the full dataset. It is recommended to work with smaller subsets of the dataset for testing purposes or to use alternative smaller datasets for experimentation.
- **Hardware**: Training 3D CNNs on video data requires significant computational resources. Ensure that you have a capable GPU or plan to run the system on a cloud-based environment like Google Colab if needed.

## Contributing

Feel free to fork the repository and submit pull requests with improvements or bug fixes. Contributions are welcome!
