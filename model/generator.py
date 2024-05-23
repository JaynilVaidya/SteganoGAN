#generator

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D,concatenate,Input,Lambda, Conv2DTranspose, LeakyReLU
from keras.utils import plot_model

def generator():
    
    ip=Input((256,256,3))
    m2=Input((64,64,1))
    m=Conv2DTranspose(16,(2,2), strides=(2,2),activation=LeakyReLU(negative_slope=0.2))(m2)
    m=xd=Conv2DTranspose(1,(2,2), strides=(2,2),activation=LeakyReLU(negative_slope=0.2))(m)
    ip2=Conv2D(64,(2,2),padding='same',activation=LeakyReLU(negative_slope=0.2))(ip)
    l1=Lambda(lambda ip2: ip2[:,:,:,:32])(ip2)
    l2=Lambda(lambda ip2: ip2[:,:,:,32:])(ip2)
    concat=concatenate([l1,m,l2],axis=-1)
    x=Conv2D(32,(3,3),padding='same',activation=LeakyReLU(negative_slope=0.2))(concat)
    x=Conv2D(16,(3,3),padding='same',activation=LeakyReLU(negative_slope=0.2))(x)
    x=Conv2D(8,(2,2),padding='same',activation=LeakyReLU(negative_slope=0.2))(x)
    op=Conv2D(3,(2,2),padding='same',activation='sigmoid')(x)

    gen=Model([ip, m2],op)

    return gen 

if __name__ == "__main__":
    gen= generator()
    gen.summary()
    #plot_model(gen, to_file='model_plot.png', show_shapes=True, show_layer_names=True)