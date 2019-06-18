# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:40:23 2019

@author: eduar
"""
#----------------------------------------------------------------------------------------
#libs que vamos usar
#----------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
from statistics import mode 
import matplotlib.pyplot as plt
import numpy as np
from datetime import date


trab = pd.read_excel("C:\\Users\\eduar\\Downloads\\GroupDatasets\\dataset.xlsx")

#----------------------------------------------------------------------------------------
#STEP 1
#----------------------------------------------------------------------------------------

# cria um gráfico que mostra os valores em falta
import missingno as msno
msno.matrix(trab,figsize=(12,5))


# preenche os dados com valores em falta com a média ou moda dos intervalos
trab= trab.fillna(trab.mean())


#----------------------------------------------------------------------------------------
#STEP 2
#----------------------------------------------------------------------------------------

# Criando novas variáveis

#  1 - total em compras por cada cliente
trab['MntTotal'] = trab['MntAcessories'] + trab['MntClothing'] + trab['MntBags'] + trab['MntAthletic'] + trab['MntShoes']

#  2 - total em produtos regulares
trab['MntRegularProds'] = trab['MntTotal'] - trab['MntPremiumProds']

#  3 - idade
trab['Age'] = date.today().year - trab['Year_Birth']

#  4 - Número total de compras por cliente
trab['Frequency'] = trab['NumCatalogPurchases'] + trab['NumStorePurchases'] + trab['NumWebPurchases']

#  5 - Média de dinheiro gasto por cliente em cada compra
trab['Avg_Mnt_Freq'] = trab['MntTotal'] / trab['Frequency']

#  6 - binario - 1: gradutation,master or PhD e 0: o resto
trab['Higher_Education'] = np.where((trab['Education'] == 'Graduation') | \
    (trab['Education'] == 'Master') | (trab['Education'] == 'PhD'), 1, 0)

#  7 - Número total de campanhas aceitas
trab['totalAcceptedCmp'] = trab['AcceptedCmp1'] + trab['AcceptedCmp2'] + trab['AcceptedCmp3'] + trab['AcceptedCmp4'] + trab['AcceptedCmp5']

#  8 - A quanto tempo é cliente
trab['AgeAsCustomer'] = trab['Custid']
for x,y in trab['Dt_Customer'].iteritems():
    y = date.today().year - y.date().year
    trab['AgeAsCustomer'] = trab['AgeAsCustomer'].set_value(x,y)

#  9 - Effort
trab['Effort'] = (trab['MntTotal'] / trab['Income']) * 100
trab['Marital_Status_High_Effort'] = trab['Marital_Status']

#----------------------------------------------------------------------------------------
#STEP 3
#----------------------------------------------------------------------------------------

# CHECAGEM DE COÊRENCIA

moneySpent = trab['MntTotal'] > 0
noFrequency = trab['Frequency'] == 0
aux  = trab['NumWebPurchases'].mean()
trab['NumWebPurchases']=np.where(moneySpent & noFrequency, aux, trab['NumWebPurchases'])

#----------------------------------------------------------------------------------------
#STEP 5
#----------------------------------------------------------------------------------------

# ESTATISTICAS DE CADA VARIAVEL
print(trab['Year_Birth'].describe())
print(trab['Income'].describe())
print(trab['Kidhome'].describe())
print(trab['Teenhome'].describe())
print(trab['Recency'].describe())
print(trab['MntAcessories'].describe())
print(trab['MntBags'].describe())
print(trab['MntClothing'].describe())
print(trab['MntAthletic'].describe())
print(trab['MntShoes'].describe())
print(trab['MntPremiumProds'].describe())
print(trab['NumDealsPurchases'].describe())
print(trab['NumWebPurchases'].describe())
print(trab['NumCatalogPurchases'].describe())
print(trab['NumStorePurchases'].describe())
print(trab['NumWebVisitsMonth'].describe())



# HISTOGRAMaS
Marital_status = sns.countplot(trab['Marital_Status'])
Education = sns.countplot(trab['Education'])
AcceptedCmp1 = sns.countplot(trab['AcceptedCmp1'])
AcceptedCmp2 = sns.countplot(trab['AcceptedCmp2'])
AcceptedCmp3 = sns.countplot(trab['AcceptedCmp3'])
AcceptedCmp4 = sns.countplot(trab['AcceptedCmp4'])
AcceptedCmp5 = sns.countplot(trab['AcceptedCmp5'])


Income = sns.distplot(trab['Income'] )



# BOXPLOTS
sns.boxplot(data=trab,x="Income",orient="v")
sns.boxplot(data=trab,x="MntAcessories",orient="v")
sns.boxplot(data=trab,x="MntBags",orient="v")
sns.boxplot(data=trab,x="MntAthletic",orient="v")
sns.boxplot(data=trab,x="MntShoes",orient="v")
sns.boxplot(data=trab,x="MntClothing",orient="v")
sns.boxplot(data=trab,x="MntPremiumProds",orient="v")

#----------------------------------------------------------------------------------------
#STEP 6
#----------------------------------------------------------------------------------------


#PCA
from sklearn.decomposition import PCA
from pandas import DataFrame

#CUMULATIVE PROPORTION OF VARIANCE EXPLAINED
prodUse = trab[['MntAcessories', 'MntClothing', 'MntBags', 'MntAthletic', 'MntShoes']]
pca = PCA(n_components = 5)
pca.fit(prodUse)
projected = pca.fit_transform(prodUse)

print('nComps', pca.components_)
print('explained' ,np.round( pca.explained_variance_,decimals = 4)*100)

var1= np.cumsum(pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.plot(var1)


#PCA NORMALIZED AND SIMPLIFIED

#Run this first
def z_score(pdusage):
    """Remove a média e normaliza-os pelo desvio padrão"""
    return (pdusage - pdusage.mean()) / pdusage.std()

pca = PCA(n_components=3)
pca.fit(prodUse.apply(z_score).T)

#loadings
loadings = DataFrame(pca.components_.T)
loadings.index = ['PC %s' % pc for pc in loadings.index + 1]
loadings.columns = ['TS %s' % pc for pc in loadings.columns + 1]
loadings

PCs = np.dot(loadings.values.T, prodUse)

marker = dict(linestyle='none', marker='o', markersize=7, color='blue', alpha=0.5)

fig, ax = plt.subplots(figsize=(7, 2.75))
ax.plot(PCs[0], PCs[1], label="Scores", **marker)
plt.grid(True)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

text = [ax.text(x, y, t) for x, y, t in
        zip(PCs[0], PCs[1]+0.5, prodUse.columns)]


#----------------------------------------------------------------------------------------
#STEP 7
#----------------------------------------------------------------------------------------


# MATRIZ DE CORRELAÇÃO
import seaborn as sns

corr = trab[['Income','Age', 'AgeAsCustomer','Avg_Mnt_Freq', 'Effort', 'Frequency', 'Teenhome', 'Kidhome', 'Recency', 'MntPremiumProds', 'MntRegularProds', 'MntTotal', 'NumDealsPurchases', 'totalAcceptedCmp']]
corr=trab[['MntAcessories', 'MntBags', 'MntClothing', 'MntAthletic', 'MntShoes']]


mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

f, ax = plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})
f.savefig('myimage.png', format='png', dpi=1200)

corr = trab.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)

continua...
