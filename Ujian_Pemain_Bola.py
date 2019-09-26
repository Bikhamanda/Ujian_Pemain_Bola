import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

databola = pd.read_csv('dataFIFA.csv')

target = []
for x, y, z in zip(databola['Age'], databola['Overall'], databola['Potential']):
    if x <= 25 and y >= 80 and z >= 80:
        target.append(1)
    else :
        target.append(0)

databola['Target'] = target 
x1 = databola[['Age', 'Overall', 'Potential']]
y1 = databola[[ 'Target' ]]

# # ====== Modelling
# == Decision Tree
modelDT = DecisionTreeClassifier()
modelDT.fit(x1, y1)

# == Kneighbors Classifier
def nilai_k():
    k=round(len(x1) **.5)
    if k % 2 == 0:
        return k+1
    else:
        return k
modelKNC = KNeighborsClassifier(
    n_neighbors=nilai_k()
)
modelKNC.fit(x1, y1)

# == Random Forest Classifier
modelRFC = RandomForestClassifier(
    n_estimators = 10
)
modelRFC.fit(x1, y1)

# # ======== Perbandingan model
# print(
#     np.mean(cross_val_score(
#     DecisionTreeClassifier(), 
#     x1, 
#     y1 ))
# ) # output : 0.8933926511781184
# print(np.mean(cross_val_score(
#     KNeighborsClassifier(n_neighbors=nilai_k()), 
#     x1, 
#     y1))
# ) # output : 0.9422749491953644
# print(np.mean(cross_val_score(
#     RandomForestClassifier(n_estimators=10), 
#     x1, 
#     y1))
# ) # output : 0.9430988081507112

pemain=pd.DataFrame([
    {'Name':'Andik Vermansyah','Age':27,'Overall':87,'Potential':90},
    {'Name':'Awan Setho Raharjo','Age':22,'Overall':75,'Potential':83},
    {'Name':'Bambang Pamungkas','Age':38,'Overall':85,'Potential':75},
    {'Name':'Cristian Gonzales','Age':43,'Overall':90,'Potential':85},
    {'Name':'Egy Maulana Vikri','Age':18,'Overall':88,'Potential':90},
    {'Name':'Evan Dimas','Age':24,'Overall':85,'Potential':87},
    {'Name':'Febri Hariyadi','Age':23,'Overall':77,'Potential':80},
    {'Name':'Hansamu Yama Pranata','Age':24,'Overall':82,'Potential':85},
    {'Name':'Septian David Maulana','Age':22,'Overall':83,'Potential':90},
    {'Name':'Stefano Lilipaly','Age':29,'Overall':88,'Potential':86}]
)

pemain['Target'] = modelKNC.predict(pemain.drop('Name',axis=1))
pemain['Target']=pemain['Target'].apply(
    lambda i: 'Target' if i==1 
    else 'Non Target'
)

print(pemain)