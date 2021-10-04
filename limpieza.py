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
missings_porcentaje.to_frame()
df_missings = pd.DataFrame(
    {'Missings': missings, 'Missings en porcentaje': missings_porcentaje}
).sort_values(by="Missings", ascending=False)
df_missings

# ### Tipos de variables

# Se puede ver que en la columna `presion_atmosferica_tarde` el tipo no es numérico

df['presion_atmosferica_tarde'].dtype

# Por lo que procedemos a reemplzar los valores no numéricos a `NaN`

# df['presion_atmosferica_tarde'].astype('float64')
try:
    df['presion_atmosferica_tarde'].astype('float64')
except ValueError as error:
    print('Error: ', error)


# Surgió un error con un valor mal cargado, asi que casteamos los valores mal cargados a `NaN`

# +
def cast_to_float(x):
    try:
        return float(x)
    except ValueError:
        return 'NaN'


df['presion_atmosferica_tarde'] = df['presion_atmosferica_tarde'].apply(cast_to_float)
df['presion_atmosferica_tarde'] = df['presion_atmosferica_tarde'].astype('float64')
# -

# ### Columnas con muchos missings

# Se puede ver que las columnas con missings mayores a 15% son

df_missings[df_missings["Missings en porcentaje"] > 15]

# Si vemos de esas columnas, en cuanta cantidad tienen `0`s en comparación con `NaN`s

cols_con_missings = df_missings[df_missings["Missings en porcentaje"] > 15].index
cant_nans = []
cant_cero = []
cant_comun = []
for col in cols_con_missings:
    cant_cero.append(df[df[col] == 0].size)
    cant_nans.append(df[df[col].isna()].size)
    cant_comun.append(df[~(df[col].isna()) & ~(df[col] == 0)].size)
df_missings_cant = pd.DataFrame(
    data={
        'Columna con missings': cols_con_missings,
        'Cantidad de NaNs': cant_nans,
        'Cantidad de ceros': cant_cero,
        'Cantidad de otros valores': cant_comun,
    }
)
df_missings_cant

# En porcentaje

100 * df_missings_cant['Cantidad de ceros'] / df_missings_cant['Cantidad de NaNs']

# Se podría asumir que, con los pocos datos que hay con valores 0 en `horas_de_sol` y `mm_evaporados_agua`, en estos casos `NaN` pueden indicar el mismo valor
#
# Por lo que procedemos a sustituirlos

columnas = ['horas_de_sol', 'mm_evaporados_agua']
for col in columnas:
    df[col].replace(np.nan, 0, inplace=True)

# ### Columnas con diferentes horarios

# Se puede observar como hay varias columnas en las que las mediciones fueron hechas temprano, y otras en la tarde.
#
# Podemos reemplazar los missings de estas columnas con el valor que deberían tener en sus horarios opuestos
#
# Vemos la relación que tienen entre sí en promedio

# +
columnas = [
    'humedad',
    'nubosidad',
    'presion_atmosferica',
    'temperatura',
    'velocidad_viendo',
]
relacion_horarios = {}
for columna in columnas:
    col_tempr = columna + '_temprano'
    col_tarde = columna + '_tarde'
    relacion_horarios[col_tempr] = (
        df[df[col_tempr].notna()][col_tempr] - df[df[col_tempr].notna()][col_tarde]
    ).mean()
    relacion_horarios[col_tarde] = (
        df[df[col_tarde].notna()][col_tarde] - df[df[col_tarde].notna()][col_tempr]
    ).mean()

pd.DataFrame(relacion_horarios, index=[0])
# -
# Entonces, con estos datos podemos llenar los faltantes en el caso de que una de las dos columnas tenga `NaN`


# +
def llenar_nans_con_horario_alterno(fila, columna, columna_alt):
    if np.isnan(fila[columna]):
        if np.isnan(fila[columna_alt]):
            return np.nan
        return relacion_horarios[columna] + fila[columna_alt]
    return fila[columna]


columnas = [
    'humedad',
    'nubosidad',
    'presion_atmosferica',
    'temperatura',
    'velocidad_viendo',
]
for columna in columnas:
    col_tempr = columna + '_temprano'
    col_tarde = columna + '_tarde'
    df[col_tempr] = df.apply(
        lambda fila: llenar_nans_con_horario_alterno(fila, col_tempr, col_tarde), axis=1
    )
    df[col_tarde] = df.apply(
        lambda fila: llenar_nans_con_horario_alterno(fila, col_tarde, col_tempr), axis=1
    )
# -

df['humedad_temprano'].isna()

# ## Conversión a variables categóricas

# De las columnas en string que tiene el dataset, las de `direccion_viento_temprano` y `direccion_viento_tarde` resultan las más categorizables, ya que se pueden traducir a ángulos cartesianos según los [puntos del compás](https://es.wikipedia.org/wiki/Puntos_del_comp%C3%A1s)

# +
puntos_del_compas = {
    'Norte': 0,
    'Nornoreste': 22.5,
    'Noreste': 45,
    'Estenoreste': 67.5,
    'Este': 90,
    'Estesureste': 112.5,
    'Sureste': 135,
    'Sursureste': 157.5,
    'Sur': 180,
    'Sursuroeste': 202.5,
    'suroeste': 225,
    'Oestesuroeste': 247.5,
    'Oeste': 270,
    'Oestenoroeste': 292.5,
    'Noroeste': 315,
}
columnas_viento = ['direccion_viento_temprano', 'direccion_viento_tarde']

for columna_viento in columnas_viento:
    for punto in puntos_del_compas.keys():
        df[columna_viento].replace(punto, puntos_del_compas[punto], inplace=True)
