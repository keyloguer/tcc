#%%
#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
"""Pandas Data"""
data = pd.read_json('final.json')

#%%
df = pd.DataFrame(data, columns = ['quantidadeDownloads',      
                                   'score',
                                   'quantidadeClassificacoes',
                                   'quantidadeComentarios',
                                   'preco',
                                   'free',
                                   'offersIap',
                                   'versaoAndroid',
                                   'desenvolvedor',
                                   'genero',
                                   'adSupported'])

'''Inicio das binarizações'''
dfVersao = pd.get_dummies(df.versaoAndroid, prefix='Versao')
dfGenero = pd.get_dummies(df['genero'])
dfFree = pd.get_dummies(df['free']).iloc[:, 1:]
dfOffersIap = pd.get_dummies(df['offersIap']).iloc[:, 1:]
dfadSupported = pd.get_dummies(df['adSupported']).iloc[:, 1:]
dfDesenvolvedor = pd.get_dummies(df['desenvolvedor'])
'''End das binarizações'''

#%%
'''Drop table'''
df = df.drop('desenvolvedor',axis=1)
df = df.drop('versaoAndroid',axis=1)
df = df.drop('free',axis=1)
df = df.drop('offersIap',axis=1)
df = df.drop('adSupported',axis=1)
df = df.drop('genero', axis=1)
'''End Drop table'''

#%%
dfQuantidadedeDownload=(df['quantidadeDownloads']-df['quantidadeDownloads'].min())/(df['quantidadeDownloads'].max()-df['quantidadeDownloads'].min())
dfQuantidadeClassificacoes=(df['quantidadeClassificacoes']-df['quantidadeClassificacoes'].min())/(df['quantidadeClassificacoes'].max()-df['quantidadeClassificacoes'].min())
dfQuantidadeComentarios=(df['quantidadeComentarios']-df['quantidadeComentarios'].min())/(df['quantidadeComentarios'].max()-df['quantidadeComentarios'].min())
dfPreco = (df['preco']-df['preco'].min())/(df['preco'].max()-df['preco'].min())
normalized_df = pd.concat([dfQuantidadedeDownload,dfQuantidadeClassificacoes,dfQuantidadeComentarios,dfPreco,df], axis=1)
print(normalized_df)

#%%
'''Concatenando todos os itens da tabela para regressão'''
jogosAndroid = pd.concat([dfVersao,dfDesenvolvedor,dfFree,dfOffersIap,dfadSupported,dfGenero,normalized_df], axis=1)
print(jogosAndroid)
"""End concatenação"""
#%%
'''Separando quantidade parecida de exemplos'''
jogosAndroidXMenorQueQuatro = jogosAndroid[jogosAndroid['score'] < 4]
jogosAndroidXMaiorQueQuatro = jogosAndroid[jogosAndroid['score'] >= 4].sample(n=300)
jogosAndroidQuatidadeProporcional = jogosAndroidXMenorQueQuatro.append(jogosAndroidXMaiorQueQuatro)
print(jogosAndroidQuatidadeProporcional)
#%%
'''Iniciando base para classificação'''
def classifier(row):
    if row['score'] >= 4:
        return 1
    elif row['score'] < 3.5:
        return 0
    else:
        return None

#%%
'''Classificando'''
dfClassification =  jogosAndroid
dfClassification['classification'] = dfClassification.apply(classifier, axis=1)
'''Classificado'''
#%%
'''Separando em quantidades iguais'''
dfClassificationGoodSample = dfClassification[dfClassification['classification'] == 1].sample(n=753)
dfClassificationBadSample = dfClassification[dfClassification['classification'] == 0]
dfClassification = dfClassificationBadSample.append(dfClassificationGoodSample)
dfClassification.drop('score', axis=1)
print(dfClassification)

#%%
'''Separando Conjunto de Classificação X e Y'''
dfClassificationY = dfClassification['classification']
dfClassificationX = dfClassification.drop('classification', axis=1)

#%%
'''Separando Conjunto de Regressão Proporcional X e Y'''
jogosAndroidQuatidadeProporcionalY = jogosAndroidQuatidadeProporcional['score']
jogosAndroidQuatidadeProporcionalX = jogosAndroidQuatidadeProporcional.drop('score', axis=1)


#%%
'''Sepraando Conjunto de Regressão com TUDO X e Y'''
jogosAndroidY = jogosAndroid['score']
jogosAndroidX = jogosAndroid.drop('score', axis=1)

#%%
'''Carregando os Algoritmos'''
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

#%%
'''Algoritimo de Regressão'''
def regression(regression):
    scores = cross_val_score(regression, jogosAndroidQuatidadeProporcionalX, jogosAndroidQuatidadeProporcionalY, cv=10)
    predict = cross_val_predict(regression, jogosAndroidQuatidadeProporcionalX, jogosAndroidQuatidadeProporcionalY, cv=10)
    mse = cross_val_score(regression, jogosAndroidQuatidadeProporcionalX, jogosAndroidQuatidadeProporcionalY, cv=10,scoring='neg_mean_squared_error')
    mae = cross_val_score(regression, jogosAndroidQuatidadeProporcionalX, jogosAndroidQuatidadeProporcionalY, cv=10,scoring='neg_mean_absolute_error')
    print ('Predict: ', predict)
    print('max: ', max(predict))
    print('min: ', min(predict))
    print('R2: ', scores.mean())
    print('MAE: ', mae.mean())
    print('MSE: ', mse.mean())
    print('RMSE: ', np.square(mse.mean()))
    plt.scatter(jogosAndroidQuatidadeProporcionalY, predict)

#%%
'''Algoritimo de Classificação'''
def classifier(classifier):
    scores = cross_val_score(classifier, dfClassificationX, dfClassificationY, cv=10)
    f1 = cross_val_score(classifier, dfClassificationX, dfClassificationY, cv=10, scoring='f1')
    precision = cross_val_score(classifier, dfClassificationX, dfClassificationY, cv=10, scoring='precision')
    recall = cross_val_score(classifier, dfClassificationX, dfClassificationY, cv=10, scoring='recall')
    predictions = cross_val_predict(classifier, dfClassificationX, dfClassificationY, cv=10)
    print ("Accuracy:", scores.mean())
    print ('F1:', f1.mean())
    print ('Precision', precision.mean())
    print ('Recall', recall.mean())
    cnf_matrix = confusion_matrix(dfClassificationY,predictions)
    plt.scatter(dfClassificationY, predictions)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()

#%%
'''Matriz de confusão'''
def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#%%
'''Iniciando Algoritimos de Regressão'''
lm = LinearRegression()
regression(lm)
#%%
neighbors = 9
knn = KNeighborsRegressor(n_neighbors=neighbors)
regression(knn)

#%%
dtr = DecisionTreeRegressor(max_depth=5)
regression(dtr)
'''Finalizando Algoritimos de Regressão'''

#%%
'''Iniciando Algoritimos de Classficação'''
dtc = DecisionTreeClassifier(random_state=0)
classifier(dtc)
#%%
neighbors2 = 9
neigh = KNeighborsClassifier(n_neighbors=neighbors2)
classifier(neigh)

#%%
lr = LogisticRegression()
classifier(lr)

#%%
gnb = GaussianNB()
classifier(gnb)
'''Finalizando Algoritimos de Classficação'''