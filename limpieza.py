# -*- coding: utf-8 -*-
# # Limpieza de datos

# ## Carga

# Empezamos importando las librerías y cargando los datasets en memoria

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import KNNImputer

df = pd.read_csv("datasets/features.csv", low_memory=False)

df2 = pd.read_csv("datasets/target.csv")

# Mezclamos los dos datasets para tener más comodidad

df = pd.merge(df, df2, on='id', how='outer')

# ## Verificando la calidad de los datos

# Queremos usar `NaN` como el valor nulo, para esto debemos uniformizar si hay otro tipo de dato faltantes

# Chequeamos que no haya celdas con `-` o espacios, usando los regexs `^-*[0-9]*\$` y `^ *\$`

regex1 = re.compile("^-*[0-9]*\$")
regex2 = re.compile("^ *\$")
df.astype(str).applymap(lambda x: bool(regex1.match(x) or regex2.match(x))).any(
    0
).to_frame("Tiene otros valores nulos")

# Vemos cuáles son las columnas que son numéricas

df[df.select_dtypes(include=[np.number]).columns]

# Exceptuando las relacionadas con la temperatura, ninguna de las columnas debería tener valores negativos

columnas_a_verificar = df.select_dtypes(include=[np.number]).columns.drop(
    ['temp_min', 'temp_max', 'temperatura_tarde', 'temperatura_temprano']
)
(df[columnas_a_verificar] < 0).any().to_frame('Menor que 0')

# Tambien verificamos si hay filas duplicadas

len(df[df.duplicated(keep=False)])

# ## Missings

# Verificamos la cantidad de nulls que hay en el datasets, y en porcentaje

missings = df.isna().sum()
missings_porcentaje = missings * 100 / len(df)
df_missings = pd.DataFrame(
    {'Missings': missings, 'Missings en porcentaje': missings_porcentaje}
).sort_values(by="Missings", ascending=False)
df_missings

# Se puede ver que las columnas con missings mayores a 15% son

df_missings[df_missings["Missings en porcentaje"] > 15]


# Cómo las columnas `nubosidad_temprano` y `nubosidad_tarde` están relacionadas entre sí, podemos usar KNN para rellenar los nulls

# +
def hashing_encoding(df, cols, data_percent=0.85, verbose=False):
    for i in cols:
        val_counts = df[i].value_counts(dropna=False)
        s = sum(val_counts.values)
        h = val_counts.values / s
        c_sum = np.cumsum(h)
        c_sum = pd.Series(c_sum)
        n = c_sum[c_sum > data_percent].index[0]
        if verbose:
            print("n hashing para ", i, ":", n)
        if n > 0:
            fh = FeatureHasher(n_features=n, input_type='string')
            hashed_features = fh.fit_transform(
                df[i].astype(str).values.reshape(-1, 1)
            ).todense()
            df = df.join(pd.DataFrame(hashed_features).add_prefix(i + '_'))

    return df.drop(columns=cols)


def knn_imputer(df):

    cat_cols = ['nubosidad_temprano', 'nubosidad_tarde']

    # Aplicamos hashing para las categoricas
    df = hashing_encoding(df, cat_cols)

    # Eliminamos variables no categóricas para imputar
    df = df.drop(columns=df.select_dtypes(include=[object]))

    # definimos un n arbitrario
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df


knn_imputer(df.head(10000)).add_suffix('_knn')
