import pandas as pd 
from io import StringIO
"""
L'idéee serait de separer le fichier en 2 la première partie contenant les 
données relative à la caméra  serparé par des , et la 2eme contenant la liste 
des vehicules separé par des ; 
Cela permettrait d'avoir un nombre fixe de colonnes car sinon cela variait en 
fonction du nombre de pawns detecté
Les header seraient  de cette première partie serait: 
     Frame, Time, Xpos, Ypos, Zpos, DetectedPawns, List_of_vehicles"

Ensuite en fonction du nombre de pawn, on pourrait checker la liste de classes 
et extraire les données qu'on voudrait '

Bref c'est une proposition

PS: y-aurait un moyen d'avoir un schema en 3d des references utilisé pour XYZ @ ...'

"""



string = 'prototype.txt'

data = pd.read_csv(string)
print(data.head())





StringData = StringIO(data.iat[2,6]) 
step = 'Vehicul.txt'
col = ['ID', 'Class', 'Xpos', 'Ypos','Zpos','X2D','Y2D','v','xdir','ydir',\
       'zdir','Xbox','Ybox','Zbox','Xext',\
       'Yext','Zext','XYZ@X','XYZ@Y','XYZ@Z',\
         'XY-Z@X','XY-Z@Y','XY-Z@Z','X-YZ@X',\
        'X-YZ@Y','X-YZ@Z','X-Y-Z@X','X-Y-Z@Y','X-Y-Z@Z',\
      '-XYZ@X','-XYZ@Y','-XYZ@Z','-XY-Z@X','-XY-Z@Y','-XY-Z@Z',\
          '-X-YZ@X','-X-YZ@Y','-X-YZ@Z','-X-Y-Z@X','-X-Y-Z@Y','-X-Y-Z@Z',\
      'X2DC', 'Y2DC','XY-Z2@X','XY-Z2@Y','X-YZ2@X','X-YZ2@Y','X-Y-Z2@X',\
          'X-Y-Z2@Y','-XYZ2@X','-XYZ2@Y','-XY-Z2@X','-XY-Z2@Y',\
              '-X-YZ2@X','-X-YZ2@Y','XYZ2@X','-X-Y-Z2@Y']
rep = pd.read_csv(StringData,sep=';',names =col ,index_col=False, lineterminator = '\\') 

print(rep.head())