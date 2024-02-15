import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

@tf.keras.utils.register_keras_serializable()
class single_headed_convolution(tf.keras.layers.Layer):                         #In case we need to use pruning it is necessary for the layer "single_headed_convolution" to inherit from tfmot.sparsity.keras.PrunableLayer additionally to tf.keras.layers.Layer   
    # Initialize all the layers inside the single headed convolution
    def __init__(self, kernel_size, num_channels, max_pool, **kwargs):
        super().__init__(**kwargs)
        self.tcn_block = []
        self.max_pool = max_pool
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.tcn_block.append(tf.keras.layers.Conv1D(out_channels, self.kernel_size, strides=1, padding="causal", dilation_rate=dilation_size))
            self.tcn_block.append(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
            self.tcn_block.append(tf.keras.layers.ReLU())
            if self.max_pool[i] == 2:
                self.tcn_block.append(tf.keras.layers.MaxPooling1D())
           
    # def get_prunable_weights(self):
    #     return self.weights
    

    # get_config() is a method inherited from tf.keras.layers.Layer that returns a dictionary containing the configuration of the layer. This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["tcn_block"] = self.tcn_block
        config["max_pool"] = self.max_pool
        config["num_levels"] = self.num_levels
        config["kernel_size"] = self.kernel_size
        return config

    def call(self, x):
        o = tf.reshape(x, (-1,120,1))
        for layer in self.tcn_block:
            o = layer(o)

        return o
    
@tf.keras.utils.register_keras_serializable()
class multi_headed_convolution(tf.keras.layers.Layer):                               #In case we need to use pruning it is necessary for the layer "multi_headed_convolution" to inherit from tfmot.sparsity.keras.PrunableLayer additionally to tf.keras.layers.Layer 
    # Initialize the single headed convolution layers
    def __init__(self, vital_signs, kernel_size, num_channels, max_pool, **kwargs):
        super().__init__(**kwargs)
        self.vital_signs = vital_signs
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.max_pool = max_pool
        self.vital_signs_list = []
        
        for i in range(self.vital_signs):
            self.vital_signs_list.append(single_headed_convolution(self.kernel_size, self.num_channels, self.max_pool))


    # def get_prunable_weights(self):
    #     return self.weights

    # get_config() is a method inherited from tf.keras.layers.Layer that returns a dictionary containing the configuration of the layer. This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["vital_signs"] = self.vital_signs
        config["vital_signs_list"] = self.vital_signs_list
        config["kernel_size"] = self.kernel_size
        config["num_channels"] = self.num_channels
        config["max_pool"] = self.max_pool
        return config

    
    def call(self, x):
        tcn_out = []
        for i, tcn_i in enumerate(self.vital_signs_list):
            o = tcn_i(x[:,:,i])
            tcn_out.append(o)
        o = tf.concat(tcn_out, axis=1)
        return o



@tf.keras.utils.register_keras_serializable()
class sensor_fusion_block(tf.keras.layers.Layer):                         #In case we need to use pruning it is necessary for the layer "single_headed_convolution" to inherit from tfmot.sparsity.keras.PrunableLayer additionally to tf.keras.layers.Layer   
    # Initialize all the layers inside the single headed convolution
    def __init__(self, kernel_size, num_channels, max_pool, head_width, **kwargs):
        super().__init__(**kwargs)
        self.tcn_block = []
        self.max_pool = max_pool
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        self.head_width = head_width
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.tcn_block.append(tf.keras.layers.Conv1D(out_channels, self.kernel_size, strides=1, padding="causal", dilation_rate=dilation_size))
            self.tcn_block.append(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
            self.tcn_block.append(tf.keras.layers.ReLU())
            if self.max_pool[i] == 2:
                self.tcn_block.append(tf.keras.layers.MaxPooling1D())
           
    # def get_prunable_weights(self):
    #     return self.weights
    

    # get_config() is a method inherited from tf.keras.layers.Layer that returns a dictionary containing the configuration of the layer. This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["tcn_block"] = self.tcn_block
        config["max_pool"] = self.max_pool
        config["num_levels"] = self.num_levels
        config["kernel_size"] = self.kernel_size
        config["head_width"] = self.head_width
        return config

    def call(self, x):
        o = tf.reshape(x, (-1,120,self.head_width))
        for layer in self.tcn_block:
            o = layer(o)

        return o

@tf.keras.utils.register_keras_serializable()
class sensor_fusion_net(tf.keras.layers.Layer):                               #In case we need to use pruning it is necessary for the layer "multi_headed_convolution" to inherit from tfmot.sparsity.keras.PrunableLayer additionally to tf.keras.layers.Layer 
    # Initialize the single headed convolution layers
    def __init__(self, vital_signs, kernel_size, num_channels, max_pool, **kwargs):
        super().__init__(**kwargs)
        self.vital_signs = vital_signs
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.max_pool = max_pool
        self.fusion_list = [2, 2, 1, 1]
        self.vital_signs_list = []
        
        for head_width in self.fusion_list:
            self.vital_signs_list.append(sensor_fusion_block(self.kernel_size, self.num_channels, self.max_pool, head_width))


    # def get_prunable_weights(self):
    #     return self.weights

    # get_config() is a method inherited from tf.keras.layers.Layer that returns a dictionary containing the configuration of the layer. This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["vital_signs"] = self.vital_signs
        config["vital_signs_list"] = self.vital_signs_list
        config["kernel_size"] = self.kernel_size
        config["num_channels"] = self.num_channels
        config["max_pool"] = self.max_pool
        config["fusion_list"] = self.fusion_list
        return config

    
    def call(self, x):
        tcn_out = []
        for i, tcn_i in enumerate(self.vital_signs_list):
            if i+1 != len(self.fusion_list):
                start_idx = sum(self.fusion_list[:i])
                end_idx = sum(self.fusion_list[:i+1])
                o = tcn_i(x[:,:,start_idx:end_idx])
                tcn_out.append(o)
            else:
                start_idx = sum(self.fusion_list[:i])
                o = tcn_i(x[:,:,start_idx:])
                tcn_out.append(o)
        o = tf.concat(tcn_out, axis=1)
        return o


def get_TCN(vital_signs,single_tcn, max_pool, num_channels, kernel_size, dense_layers, output_size, sensor_fusion):
    print("USING INPUT SIZE FOR FREQ 2")


    #input layer necessary for the model to know the shape of the input which allows the model to know the shape of the weights, biases, and activations at each layer
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(120,vital_signs))])

    # If flag single_tcn is set to true, the model will use a single TCN that takes all the vital signs as input (i.e with input shape (120, vital_signs))
    if single_tcn:
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            model.add(tf.keras.layers.Conv1D(out_channels, kernel_size, strides=1, padding="causal", dilation_rate=dilation_size))
            model.add(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
            model.add(tf.keras.layers.ReLU())
            if max_pool[i] == 2:
                model.add(tf.keras.layers.MaxPooling1D())
    elif sensor_fusion:
        model.add(sensor_fusion_net(vital_signs, kernel_size, num_channels, max_pool))
    # If flag single_tcn is set to false, the model will use "vital_signs" TCNs where each TCN takes only one channel as input (i.e each TCN will have input shape (120, 1))
    else:
        model.add(multi_headed_convolution(vital_signs, kernel_size, num_channels, max_pool))

    # Flatten the concatenated output of the TCNs so that it can be fed to the dense layers
    model.add(tf.keras.layers.Flatten())

    # Dense block
    for dense_out in dense_layers:
        model.add(tf.keras.layers.Dense(dense_out))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.05))

    #output Dense
    model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
    
    return model