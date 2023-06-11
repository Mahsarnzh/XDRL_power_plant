#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    11-Jun-2023 19:51:13

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    state = keras.Input(shape=(5,))
    CriticStateFC1_300 = layers.Dense(300, name="CriticStateFC1_300_")(state)
    CriticRelu1 = layers.ReLU()(CriticStateFC1_300)
    CriticStateFC2_300 = layers.Dense(300, name="CriticStateFC2_300_")(CriticRelu1)
    CriticRelu2 = layers.ReLU()(CriticStateFC2_300)
    output = layers.Dense(1, name="output_")(CriticRelu2)

    model = keras.Model(inputs=[state], outputs=[output])
    return model
