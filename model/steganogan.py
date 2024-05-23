#stegenogan

import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

def steganogan_model(gen,decoder):
        
    gen.trainable=True
    cover=Input((256,256,3))
    msg=Input((64,64,1))
    stegimg=gen([cover, msg])
    m=decoder(stegimg)

    steganogan=Model([cover, msg],m)
    steganogan.compile(optimizer='adam',loss=['mse'],metrics=['accuracy'])
    
    return steganogan


if __name__ == "__main__":
    
    from generator import generator
    from discriminator import discriminator
    from decoder import decoder_model
    gen=generator()
    decoder=decoder_model()
    steganogan=steganogan_model(gen,decoder)
    steganogan.summary()
    #plot_model(steganogan, to_file='model_plot.png', show_shapes=True, show_layer_names=True)