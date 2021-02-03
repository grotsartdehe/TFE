import pandas as pd 
import re
from RadarGen import *
from Search import *
text_file = open("01_gt.txt", "r")
 
#read whole file to a string
data = text_file.read()
 
#close file
text_file.close()

"""
doc_string = re.split('[a-z,:,;,=,-,@]+', data, flags = re.IGNORECASE)
print(doc_string)
"""


"""
names = ['Frame', 'T:', 'Time', 'ff', 'Pos', 'ffg', 'Position' , 'Detected', 'Pawns' ,'hfhe' ,'Class', '[List of classes']
doc = pd.read_csv(data,delimiter=names )

doc.info()
doc.shape()
"""

Z1,Z2 = RadarGen(0,[30,5],[35,5],[np.pi/3,6],[np.pi/7,np.pi/8])
res = Searchdv(Z1,1)
resangle= Searchangle(Z2,1)

