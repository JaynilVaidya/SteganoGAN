#discriminator

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LeakyReLU, Flatten, Conv2D, Input
from keras.utils import plot_model


def discriminator():
    
    inp=Input((256,256,3))
    xd=Conv2D(64,(2,2),padding='same',activation=LeakyReLU(negative_slope=0.2))(inp)
    xd=Conv2D(32,(2,2),padding='same',activation=LeakyReLU(negative_slope=0.2))(xd)
    xd=Conv2D(16,(2,2),padding='same',activation=LeakyReLU(negative_slope=0.2))(xd)
    xd=Flatten()(xd)
    xd=Dense(128,activation=LeakyReLU(negative_slope=0.2))(xd)
    xd=Dense(128,activation=LeakyReLU(negative_slope=0.2))(xd)
    opp=Dense(1,activation='softmax')(xd)

    disc=Model(inp,opp)

    disc.compile(optimizer='rmsprop',metrics=['accuracy'],loss='binary_crossentropy')
    
    return disc

if __name__ == "__main__":
    
    disc= discriminator()
    disc.summary()
    #plot_model(disc, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    