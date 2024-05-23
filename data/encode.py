import codecs
import numpy as np

def text_encode(code,shp):
  code = codecs.encode(code, 'rot_13')
  lst=np.zeros(8)
  
  for ch in code:  
    lst=np.append(lst,np.array(list(map(int,format(ord(ch), '08b')))))
  lst.resize(shp)
  
  return lst