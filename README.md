# prediction-of-sepsis-onset
This is the work that I did for my Bachelor's Thesis in collaboration with my supervisors Dr. Kanika Dheman and Marco Giordano.

In this repository you can find a pdf giving detailed explanations about the project but nonetheless, in this README I will provide a short summary of my work.

### Dataset
The Dataset used was the HiRID [https://physionet.org/content/hirid/1.1.1/] a high time-resolution ICU dataset.

### Summary

The goal of this project was to build a machine learning model that would take advantage of the high time-resolution of the HiRID dataset in order to create a lightweight model performing similarly to (much heavier) state-of-the-art models. This was with the hope of using our model for on-device inference.


### Structure of the Project

The project is divided in two. In the first part we used pytorch develop the model and preprocess the data. In the second part I worked on developing the model in Tensorflow and integrating it with the pytorch data preprocessing pipeline. All the files related to the Tensorflow implementation start with "tf_".

### Model architecture

The model used for this project is a temporal convolutional network (TCN) which is basically a 1D CNN with dilation. For an intuitive explanation on how TCN works, you can check out this link: https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4




