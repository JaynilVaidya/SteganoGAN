#gan 

import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

def gan_model(gen,disc):
    
    disc.trainable = False
    cover=Input((256,256,3))
    msg=Input((64,64,1))
    gen=gen([cover, msg])
    op=disc(gen)

    gan=Model([cover, msg],op)
    gan.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    return gan

if __name__ == "__main__":
    
    from discriminator import discriminator
    from generator import generator
    gen=generator()
    disc=discriminator()
    gan=gan_model(gen,disc)
    gan.summary()
    #plot_model(gan, to_file='model_plot.png', show_shapes=True, show_layer_names=True)