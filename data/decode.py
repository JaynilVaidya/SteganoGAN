import codecs
import numpy as np

def text_decode(code_matrix):
    
  code_matrix = code_matrix.flatten()
  code_matrix=np.round_(code_matrix, decimals=0).astype('int16')
  final = []
  lst = code_matrix.tolist()
  
  for i in range(0,len(lst),8):
    c = lst[i:i+8]
    if c==[0,0,0,0,0,0,0,0]:
      final.append('00100000')
    else: 
      str_ = ''.join(map(str,c))
      final.append(str_)
  code = ""
  
  for i in range(len(final)):
    code = code + chr(int(final[i],2))
  code = codecs.decode(code,'rot_13')
  
  return code