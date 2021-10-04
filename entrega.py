# -*- coding: utf-8 -*-
# # TP: Cuidado! Lluvia de Hamburgesas - Primera Entrega

# ## Introducción

# En el reporte a continuación, haremos un análisis de lo que se llevó a cabo en el trabajo de limpieza, análisis y conclusiones sobre los datos de la Lluvia de Hamburguesas
#
# Dados los dos links de descarga de los CSVs, los descargamos en la carpeta `datasets/` para poder trabajar localmente con los archivos

# Empezamos por cargar las librerías que se van a usar

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# En el princpio se optó por mantener los dos datasets en el mismo DataFrame, para tener mas comodidad

# +
df = pd.read_csv("datasets/features.csv", low_memory=False)
df2 = pd.read_csv("datasets/target.csv")

df = pd.merge(df, df2, on='id', how='outer')
# -

# Las columnas que manejaremos serán entonces

df.columns.to_frame("")


# ## Limpieza de datos

# Apenas importamos los datos, hicimos un análisis de que los datasets sean consistentes con que sus valores nulos sean `NaN`, y que no contengan valores inválidos
#
# Este análisis extensivo puede verse en [limpieza.py](limpieza.py), a continación se marcarán los puntos más importantes

# ### Tipos de variables

# Algo a destacar de dicho análisis es que la columna `presion_atmosferica_tarde` tenía datos inválidos, por lo que se procedió a convertir sus valores a numéricos, rellenando `NaN` cuando no se pueda

# +
def cast_to_float(x):
    try:
        return float(x)
    except ValueError:
        return 'NaN'


df['presion_atmosferica_tarde'] = df['presion_atmosferica_tarde'].apply(cast_to_float)
df['presion_atmosferica_tarde'] = df['presion_atmosferica_tarde'].astype('float64')
# -

# ### Missings

# Verificamos que el dataset tenía una gran cantidad de missings

missings = df.isna().sum()
missings_porcentaje = missings * 100 / len(df)
missings_porcentaje.to_frame()
df_missings = pd.DataFrame(
    {'Missings': missings, 'Missings en porcentaje': missings_porcentaje}
).sort_values(by="Missings", ascending=False)
df_missings

# ### Columnas con diferentes horarios

# Se puede observar como hay varias columnas en las que las mediciones fueron hechas temprano, y otras en la tarde.
#
# Si calculamos la relación entre las columnas, podemos reemplazar los datos faltantes con sus horarios opuestos

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


# Entonces, teniendo la relación que tienen entre sí en promedio, podemos rellenar los casos que se tenga un dato de los dos

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

# ### Conversión a variables categóricas

# De las columnas en string que tiene el dataset, las de `direccion_viento_temprano` y `direccion_viento_tarde` resultan las más categorizables, ya que se pueden traducir a ángulos cartesianos según los [puntos del compás](https://es.wikipedia.org/wiki/Puntos_del_comp%C3%A1s)
#
# Por lo que se procedió a categorizarla de acuerdo a eso

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
        df[columna_viento+'_num'] = df[columna_viento].replace(punto, puntos_del_compas[punto])
# -

# ## Análisis de los datos

# En esta seccion buscamos responder la pregunta: **Que relacion hay entre los features y el target?**
#
# Este análisis extensivo puede verse en [relaciones.py](relaciones.py), a continación se marcarán los puntos más importantes.

# ### Analisis Inicial

# En principio queremos saber : **Cual es la probabilidad de que lluevan hamburguesas?** para luego poder tomar ese valor de referencia.
#
# Aparte de eso, hacemos algunas transformaciones que seran de utilidad.

df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente'],inplace=True)

df['llovieron_hamburguesas_al_dia_siguiente_n'] = df.replace({'llovieron_hamburguesas_al_dia_siguiente': {'si': 100, 'no': 0}})['llovieron_hamburguesas_al_dia_siguiente']

burger_mean = df['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
burger_mean

# ### Analisis de features numericas

num_labels = ["horas_de_sol","humedad_tarde","mm_lluvia_dia","nubosidad_tarde"]


def cmpNumeric(label):
    plt.figure(dpi=150)
    plt.title(label)
    sns.boxplot(
        data=df,
        y=label,
        x='llovieron_hamburguesas_al_dia_siguiente',
        palette=['#D17049', "#89D15E"],
        showfliers=False,
    )
    plt.ylabel(label)
    plt.show()


for label in num_labels:
    cmpNumeric(label)

# #### Filtros en variables numericas

# Notamos ciertos puntos de inflexion en los graficos previos, a partir de ello nos preguntamos : **Cual es la probabilidad de que llueva pasado ese punto?**.

# +
df_num = pd.DataFrame(
    {
        'filtro':[
            "promedio normal",
            "horas_de_sol < 7",
            "humedad_tarde > 60",
            "mm_lluvia_dia is Nan",
            "nubosidad_tarde > 7",
        ],
        '%_llovieron_hamburguesas':[
            burger_mean,
            df[ (df['horas_de_sol']<7) ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[ (df['humedad_tarde']>60) ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[ (df['mm_lluvia_dia'].isna()) ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[ (df['nubosidad_tarde'] > 7) ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
        ]
    }
).sort_values(by="%_llovieron_hamburguesas", ascending=False)

plt.figure(dpi=150, figsize=(3, 2))
sns.barplot( data=df_num, y='filtro', x='%_llovieron_hamburguesas')
plt.axvline(burger_mean, 0, 1, color="#000")
plt.show()
df_num
# -

# ## Conclusiones

# ## Baseline
