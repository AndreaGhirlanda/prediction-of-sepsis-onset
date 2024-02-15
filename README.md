# Prediction of Sepsis Onset
This is the work that I did for my Bachelor's Thesis in collaboration with my supervisors Dr. Kanika Dheman and Marco Giordano.

In this repository you can find a pdf giving detailed explanations about the project.

### Dataset
The Dataset used was the [HiRID](https://physionet.org/content/hirid/1.1.1/) a high time-resolution ICU dataset.

### Summary

The primary objective of this project was to develop a machine learning model leveraging the high temporal resolution of the HiRID dataset. The aim was to create a lightweight model that could perform comparably to existing state-of-the-art models, which are often much heavier. The ultimate goal was to deploy the model for on-device inference.

### Structure of the Project

The project initially started with PyTorch. Consequently, the entire data preprocessing pipeline and model development were implemented using PyTorch. However, as the project progressed, there was a need to transition to TensorFlow. This transition was motivated by the desire to leverage the TensorFlow Lite library for model quantization and export, essential for on-device inference.

As a result, the project's structure reflects this dual-library approach. The "tf" folder contains all files related to TensorFlow integration, model quantization, and training and testing of TensorFlow models. The "models" folder contains code pertinent to the architecture of the models, irrespective of the library used. Finally, the "helper" folder encompasses functions related to dataset generation, metric calculation, and training and testing of PyTorch models.

The PyTorch implementation is accessible through the "main.py" file, while the TensorFlow implementation can be accessed through a separate main file called "tf_main.py".

### Model architecture

The chosen model architecture for this project is a Temporal Convolutional Network (TCN), essentially a 1D Convolutional Neural Network (CNN) with dilation. TCN's ability to capture long-range dependencies in sequential data makes it an ideal choice for modeling temporal data such as that found in ICU datasets. 




