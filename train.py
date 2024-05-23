import numpy as np
from model.generator import generator
from model.discriminator import discriminator
from model.decoder import decoder_model
from model.gan import gan_model
from model.steganogan import steganogan_model
import cv2 as cv2

import warnings
warnings.filterwarnings("ignore")


disc=discriminator()
gen=generator()
decoder=decoder_model()
gan=gan_model(gen,disc)
steganogan=steganogan_model(gen,decoder)


X_train = np.load('C:/Users/Jaynil/Github repos/SteganoGan/data/processed/new_data2000.npz.npy')

epochs=50
batch_size=32
batches=X_train.shape[0]//batch_size
a=[]
b=[]

code=np.random.randint(0,2,size=(1,64,64,1))
code=np.resize(code,(batch_size,64,64,1))

for e in range(epochs):
  for mb in range(batches):
    realimgs=X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

    #training decoder
    decoder.trainable=True
    stegimgs=gen.predict([realimgs, code])
    loss3=decoder.train_on_batch(stegimgs, code, return_dict=True)

    #training steganogan
    decoder.trainable=False
    gen.trainable=True
    loss4=steganogan.train_on_batch([realimgs,code],code,return_dict=True)

    #training disc
    disc.trainable=True
    fakeimgs=gen.predict([realimgs,code])
    x=np.concatenate([realimgs,fakeimgs])
    y1=np.zeros(batch_size*2)
    y1[:batch_size]=0.9
    loss1=disc.train_on_batch(x,y1,return_dict=True)

    #training gan
    disc.trainable=False
    gen.trainable=True
    y2=np.full(batch_size,0.9)
    loss2=gan.train_on_batch([realimgs,code],y2,return_dict=True)
    
    print(f"Epoch {e}/{epochs}, Batch {mb}/{batches}:")
    print(f"  Discriminator Loss: {loss1['loss']:.4f}, GAN Loss: {loss2['loss']:.4f}")
    print(f"  Decoder Loss: {loss3['loss']:.4f}, SteganoGAN Loss: {loss4['loss']:.4f}")
    print(f"  Decoder Accuracy: {loss3['accuracy']:.4f}, Discriminator Accuracy: {loss1['accuracy']:.4f}",end='\n')
    

#printing images during training

#     if mb%50==0:
#       org=X_train[np.random.randint(0, X_train.shape[0], size=1)]
#       stegimgg=gen.predict([org, code2])
#       print('Real Image\n')
#       plt.imshow((org[0]*255).astype(np.uint8))
#       plt.show()
#       print('Steg Image\n')
#       plt.imshow(cv2.cvtColor(stegimgg[0], cv2.COLOR_BGR2RGB ))
#       plt.show()
#       gc.collect()
