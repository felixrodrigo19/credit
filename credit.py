"""Treinamento de ML para analise financeiro"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("credit-data.csv")

### conhecendo e tratando o df
df.describe()
df.head(5)
df.isnull().sum()
df.loc[df['age'] < 0]


plt.hist([df['age']])


age_mean = df['age'][df['age'] > 0].mean()
df.update(df['age'].fillna(age_mean))
df.loc[df['age'].isnull()] = age_mean

# Separa previsores(X) da classe(y)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Escalonamente de valores
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Dividir em dados de treino e teste
x_train, x_test, y_train, y_test = tts(X, y, test_size=0.30, random_state=0)


# testes de performance de ML
def predict(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=10)
    sucess_rate = np.mean(scores)
    return sucess_rate

all_results = {}
# Treinamento do modelo
model = GaussianNB()
all_results['GaussianNB'] = predict('GaussianNB', model, x_train, y_train)

model = RandomForestClassifier()
all_results['RandomForestClassifier'] = predict('RandomForestClassifier', model, x_train, y_train)

model = KNeighborsClassifier()
all_results['KNeighborsClassifier'] = predict('KNeighborsClassifier', model, x_train, y_train)


for key, value in all_results.items():
    print(f"Modelo: {key} - Taxa de acerto: {value}")

# Métrica
#result = model_GaussianNB.predict(x_test)
#score = accuracy_score(result, y_test)
