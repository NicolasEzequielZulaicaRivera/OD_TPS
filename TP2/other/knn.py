# -*- coding: utf-8 -*-
# # KNN

# ### Imports

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump, load

from sklearn.neighbors import KNeighborsClassifier

from utilities import score

# ### Codigo a correr

# **Run :** Entrenar Nuevamente o Cargar Entrenado

runSimple = False
runValidated = False
runDouble = False

# **Save :** Guardar Modelo (pisa anterior)

saveSimple = True
saveValidated = True
saveDouble = True

# ### Dataset

from preprocessing import reemplazarNulls,reemplazarCategoricas,reemplazarFechas,targetBooleano
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df_feat = pd.read_csv("datasets/train_features.csv", low_memory=False)
df_targ = targetBooleano( pd.read_csv("datasets/train_target.csv") )

reemplazarNulls(df_feat , inplace=True)
reemplazarCategoricas(df_feat , inplace=True)
reemplazarFechas(df_feat , inplace=True)

df_feat.info()


# ## Entrenamiento

# Definimos una función auxiliar para separar el dataset en train y development

def get_train_dev(normalized=False):
    y = df_targ
    if (normalized):
        normalizer = preprocessing.Normalizer()
        X = normalizer.fit_transform(df_feat)
    else:
        X = df_feat.copy()

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.5, random_state=0)
    return X_train, X_dev, y_train, y_dev


# Hacemos un entrenamiento inicial con parámetros default para verificar

# +
X_train, X_val_dev, y_train, y_val_dev = get_train_dev()

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val_dev)
print('Accuracy: ', np.sum(y_val_dev == y_pred) / len(y_val_dev))
# -

# Viendo que la accuracy no es muy alta, probamos cambiando los hiperparámetros

# ### Búsqueda de hiperparámetros

# +
metrics = []
for n in range(1, 200, 5):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val_dev)
    metrics.append((n, (y_val_dev == y_pred).sum()))

# ploteamos
df_metrics = pd.DataFrame(metrics, columns=['cant_vecinos', 'correctos'])
ax = df_metrics.plot(
    x='cant_vecinos', y='correctos', title='Variando k (cantidad de vecinos)'
)
ax.set_ylabel("Cantidad de aciertos")
plt.show()
# -

df_metrics.iloc[df_metrics['correctos'].idxmax()]

# Se ve que la métrica llega a un máximo con `31` vecinos y luego se aplana

# ### Normalizando los inputs

# +
X_train, X_val_dev, y_train, y_val_dev = get_train_dev(normalized=True)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val_dev)
print('Accuracy: ', np.sum(y_val_dev == y_pred) / len(y_val_dev))
# -

# Se observa que la accuracy mejoró un poco con el parámetro por default, veamos los diferentes números de neighbors

# +
metrics = []
for n in range(1, 200, 5):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val_dev)
    metrics.append((n, (y_val_dev == y_pred).sum()))

# ploteamos
df_metrics = pd.DataFrame(metrics, columns=['cant_vecinos', 'correctos'])
print('mejor puntaje: ', max(df_metrics.correctos), 'correctos')
ax = df_metrics.plot(
    x='cant_vecinos', y='correctos', title='Variando k (cantidad de vecinos)'
)
ax.set_ylabel("Cantidad de aciertos")
plt.show()
# -

df_metrics.iloc[df_metrics['correctos'].idxmax()]

# Nuevamente se observa que llega a un máximo con `11` vecinos y luego disminuye y se aplana

# ### Evaluación final

# Con los dos datos que obtuvimos, probamos seteando `11` neighbors, y normalizando los inputs

# +
X_train, X_val_dev, y_train, y_val_dev = get_train_dev(normalized=True)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val_dev)
print('Accuracy: ', np.sum(y_val_dev == y_pred) / len(y_val_dev))
# -

# ## Conclusión

# Luego de una búsqueda de los mejores parámetros, los que fueron utilizados para llegar a los mejores resultados posibles fueron de `11` neighbors y normalizando los inputs del dataset.
#
# Por lo que se logró llegar a una accuracy de `79.30%`
#
#
