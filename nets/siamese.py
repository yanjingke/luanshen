import keras
from keras.layers import Input,Dense,Conv2D
from keras.layers import MaxPooling2D,Flatten,Lambda
from keras.models import Model
import keras.backend as K
import os
import numpy as np
from PIL import Image
from keras.optimizers import SGD
from nets.vgg import VGG16
from nets.mobilenetV3_large import MobileNetv3_large

 
def siamese(input_shape):
    VGG_model = VGG16(input_shape)
    # densenet_model =MobileNetv3_large(input_shape)
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    encoded_image_1 =  VGG_model(input_image_1)
    encoded_image_2 =  VGG_model(input_image_2)

    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    out = Dense(512,activation='relu')(l1_distance)
    out = Dense(1,activation='sigmoid')(out)

    model = Model([input_image_1,input_image_2],out)
    return model
