#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    11-Jun-2023 19:51:12

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    state = keras.Input(shape=(5,))
    ActorStateFC1_300 = layers.Dense(300, name="ActorStateFC1_300_")(state)
    ActorRelu1 = layers.ReLU()(ActorStateFC1_300)
    ActorStateFC2_300 = layers.Dense(300, name="ActorStateFC2_300_")(ActorRelu1)
    ActorRelu2 = layers.ReLU()(ActorStateFC2_300)
    action = layers.Dense(6, name="action_")(ActorRelu2)
    RepresentationSoftMax = layers.Softmax()(action)

    model = keras.Model(inputs=[state], outputs=[RepresentationSoftMax])
    return model
