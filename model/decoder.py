#decoder

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input
from keras.utils import plot_model

def decoder_model():
    
    #generator.trainable=False
    stegimg=Input((256,256,3))
    x=Conv2D(128,(3,3),padding='same',strides = (1,1), activation='relu')(stegimg)
    x=Conv2D(128,(3,3),padding='same',strides = (2,2), activation='relu')(x)
    x=Conv2D(64,(2,2),padding='same',strides = (1,1), activation='relu')(x)
    x=Conv2D(64,(3,3),padding='same',strides = (2,2), activation='relu')(x)
    x=Conv2D(32,(2,2),padding='same',strides = (1,1), activation='relu')(x)
    op=Conv2D(1,(3,3),padding='same',activation='sigmoid')(x)


    decoder=Model(stegimg,op)
    decoder.compile(loss=['mse'],optimizer='rmsprop',metrics=['accuracy'])

    return decoder


if __name__ == "__main__":
    
    decoder=decoder_model()
    decoder.summary()
    #plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)