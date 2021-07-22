# data analysis and wrangling
import pandas as pd
import numpy as np

# Visualização
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Entrada de dados
treino_df = pd.read_csv('entrada/train.csv')
teste_df = pd.read_csv('entrada/test.csv')
combinar = [treino_df, teste_df]
# Imprime dados base para verificação


treino_df = treino_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
teste_df = teste_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
combinar = [treino_df, teste_df]  # remoção de ticket, cabine e embarque

for dataset in combinar:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#print(pd.crosstab(treino_df['Title'], treino_df['Sex']))#lista por sexo
# print("\n"*3) #pula 3 linhas
for dataset in combinar:

#Padronização de boas práticas/visualização
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Raro')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(treino_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())#Lista Sobreviventes por tipo
# print("\n"*3)

#Imprimir sobreviventes por títulos
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Raro": 5}
for dataset in combinar:
    dataset['Title'] = dataset['Title'].map(title_mapping) #substituindo títulos por números
    dataset['Title'] = dataset['Title'].fillna(0) #preenchendo quem não tem títulos com 0
# treino_df.head() #verifica cabeça de lista

treino_df = treino_df.drop(['Name', 'PassengerId'], axis=1)
teste_df = teste_df.drop(['Name'], axis=1)
combinar = [treino_df, teste_df]  # Atualiza fusão depois de remover NAME e PASSENGER ID
# print(treino_df.shape, teste_df.shape)# verifica indice


for dataset in combinar:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)  # Substitui sexo por boolean

# treino_df.head() #verifa sexo boolean

advinhaidade = np.zeros((2, 3))  # Cria um array no formato para fazer a media das idades desonhecidas

for dataset in combinar:
    for i in range(0, 2):
        for j in range(0, 3):
            advinha_df = dataset[(dataset['Sex'] == i) & \
                                 (dataset['Pclass'] == j + 1)]['Age'].dropna()  # remove quem não tem idade do dataset

            idadeadvinha = advinha_df.median()  # pega a media da idade

            advinhaidade[i, j] = int(idadeadvinha / 0.5 + 0.5) * 0.5  # faz uma variação para preencher idades vazias

    for i in range(0, 2):#0 - 1
        for j in range(0, 3):#1 -2 -3
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = advinhaidade[
                i, j]  # localiza a idade de acordo com a classe e o sexo

    dataset['Age'] = dataset['Age'].astype(int)

treino_df['AgeBand'] = pd.cut(treino_df['Age'], 8) #criando uma coluna que irá se partir em 8 para calcular média das idades
#print(treino_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))#imprime lista por idade para ajudar a decidir numero de colunas

#mapa de grupos de idades. Exemplo: Grupo 3: é quem tem entre 31 e 40 anos.
for dataset in combinar:
    dataset.loc[dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6
    dataset.loc[dataset['Age'] > 70, 'Age'] = 7
treino_df = treino_df.drop(['AgeBand'], axis=1) #Apaguei o coluna em AgeBand, só foi para mapear as idades.
combinar = [treino_df, teste_df]

for dataset in combinar:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #criando o tamanho da família de acordo com o número de filhos e parceiros.

# print(treino_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)) #imprimir o tamanho das familias
teste_df['Fare'].fillna(teste_df['Fare'].dropna().median(), inplace=True)
treino_df['FareBand'] = pd.qcut(treino_df['Fare'], 8) #criando uma coluna que irá se partir em 8 para calcular média de all inclusive
# print(treino_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))#banda de fare para avaliação

for dataset in combinar:
    dataset.loc[dataset['Fare'] <= 7.75, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.75) & (dataset['Fare'] <= 7.91), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 9.841), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 9.841) & (dataset['Fare'] <= 14.454), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 24.479), 'Fare'] = 4
    dataset.loc[(dataset['Fare'] > 24.479) & (dataset['Fare'] <= 69.488), 'Fare'] = 5
    dataset.loc[(dataset['Fare'] > 69.488) & (dataset['Fare'] <= 512.329), 'Fare'] = 6
    dataset.loc[dataset['Fare'] > 512.329, 'Fare'] = 7
    dataset['Fare'] = dataset['Fare'].astype(int)  # definição de Fare range como inteiro

treino_df = treino_df.drop(['FareBand'], axis=1)
combinar = [treino_df, teste_df]

X_treino = treino_df.drop("Survived", axis=1)
Y_treino = treino_df["Survived"]
X_teste = teste_df.drop("PassengerId", axis=1).copy()
# print(X_treino.shape, Y_treino.shape, X_teste.shape)


# Support Vector Machines - ele calcula a média de sobreviventes de acordo com vetores de coluna

svc = SVC()
svc.fit(X_treino, Y_treino)
Y_pred = svc.predict(X_teste)
acc_svc = round(svc.score(X_treino, Y_treino) * 100, 2)
#print(acc_svc)
# KNN
knn = KNeighborsClassifier(n_neighbors=3)  # verifica a proximidade das pessoas
knn.fit(X_treino, Y_treino)
Y_pred = knn.predict(X_teste)
acc_knn = round(knn.score(X_treino, Y_treino) * 100, 2)
#print(acc_knn)
modelo = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN'],
    'Score': [acc_svc, acc_knn]})
print(modelo.sort_values(by='Score', ascending=False))
resultado = pd.DataFrame({
    "PassengerId": teste_df["PassengerId"],
    "Survived": Y_pred,
})
resultado.to_csv('saida/resultado.csv', index=False)
print(pd.DataFrame({
    "PassengerId": teste_df["PassengerId"], "Sex": teste_df["Sex"],
    "Survived": Y_pred,
}).groupby(['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=True))

# print(treino_df.sort_values(by='Fare', ascending=True))#imprime lista de treino
