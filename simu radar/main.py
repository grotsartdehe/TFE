import pandas as pd 
import re
from RadarGen import *
from Search import *
from io import StringIO

string = 'prototype.txt'

data = pd.read_csv(string)
data.head()




StringData = StringIO(data.iat[2,6]) 
step = 'Vehicul.txt'
col = ['ID', 'Class', 'Xpos', 'Ypos','Zpos','X2D','Y2D','v','xdir','ydir','zdir','Xbox','Ybox','Zbox','Xext',\
       'Yext','Zext','XYZ@X','XYZ@Y','XYZ@Z','XY-Z@X','XY-Z@Y','XY-Z@Z','X-YZ@X','X-YZ@Y','X-YZ@Z','X-Y-Z@X','X-Y-Z@Y','X-Y-Z@Z',\
      '-XYZ@X','-XYZ@Y','-XYZ@Z','-XY-Z@X','-XY-Z@Y','-XY-Z@Z','-X-YZ@X','-X-YZ@Y','-X-YZ@Z','-X-Y-Z@X','-X-Y-Z@Y','-X-Y-Z@Z',\
      'X2DC', 'Y2DC','XY-Z2@X','XY-Z2@Y','X-YZ2@X','X-YZ2@Y','X-Y-Z2@X','X-Y-Z2@Y','-XYZ2@X','-XYZ2@Y','-XY-Z2@X','-XY-Z2@Y','-X-YZ2@X','-X-YZ2@Y','XYZ2@X','-X-Y-Z2@Y']
rep = pd.read_csv(StringData,sep=';',names =col ,index_col=False, lineterminator = '\\') 
print(rep.shape)

"""
names = ['Frame', 'T:', 'Time', 'ff', 'Pos', 'ffg', 'Position' , 'Detected', 'Pawns' ,'hfhe' ,'Class', '[List of classes']
doc = pd.read_csv(data,delimiter=names )

doc.info()
doc.shape()
"""
"""
Z1,Z2 = RadarGen(0,[30,5],[35,5],[np.pi/3,6],[np.pi/7,np.pi/8])
res = Searchdv(Z1,1)
resangle= Searchangle(Z2,1)

"""