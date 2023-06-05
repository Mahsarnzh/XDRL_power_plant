#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    01-Jun-2023 16:56:11

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input_1 = keras.Input(shape=(5,))
    fc_1 = layers.Dense(256, name="fc_1_")(input_1)
    relu_body = layers.ReLU()(fc_1)
    fc_body = layers.Dense(256, name="fc_body_")(relu_body)
    body_output = layers.ReLU()(fc_body)
    output = layers.Dense(1, name="output_")(body_output)

    model = keras.Model(inputs=[input_1], outputs=[output])
    return model
