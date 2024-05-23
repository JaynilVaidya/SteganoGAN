from tensorflow import keras
import numpy as np
import gc
import cv2
import numpy as np
from data.encode import text_encode
from data.decode import text_decode

img=cv2.imread('/data/raw/sample_img.jpg')
img = np.array(img)
img=img/255

gen=keras.models.load_model('data/processed/gen.h5')
decoder=keras.models.load_model('data/processed/decoder.h5')

msg="Sample message to be concealed"
shape_arr = (64,64,1)
code_matrix = text_encode(msg,shape_arr)

steg_img=gen([img.reshape((1,256,256,3)),code_matrix.reshape((1,64,64,1))])
decoded_matrix=decoder(steg_img)

decoded_message=text_decode(decoded_matrix)

cv2.imwrite('data/processed/generated_image.jpg') 
with open('data/processed/decoded_message', "w") as file: file.write(decoded_message)





