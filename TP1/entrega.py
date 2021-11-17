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
missings_totales_iniciales = df.isna().sum(axis=1).sum()
missings_porcentaje = missings * 100 / len(df)
missings_porcentaje.to_frame()
df_missings = pd.DataFrame(
    {'Missings': missings, 'Missings en porcentaje': missings_porcentaje}
).sort_values(by="Missings", ascending=False)
df_missings

# ### Columnas con diferentes horarios

# Se puede observar como hay varias columnas en las que las mediciones fueron hechas temprano, y otras en la tarde.
#
# De estos datos, hay algunas ocasiones en las que había el dato temprano y en la tarde había un missing, y viceversa.

# +
plt.figure(dpi=150)
labels = ['Falta temprano', 'Falta tarde']
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()

i = -2
for columna in [
    'humedad',
    'nubosidad',
    'presion_atmosferica',
    'temperatura',
    'velocidad_viendo',
]:
    datos = {
        'Falta temprano': len(
            df[(df[columna + '_temprano'].isna()) & (df[columna + '_tarde'].notna())]
        ),
        'Falta tarde': len(
            df[(df[columna + '_temprano'].notna()) & (df[columna + '_tarde'].isna())]
        ),
    }
    ax.bar(x + i * width / 5, datos.values(), width / 5, label=columna)
    i += 1

ax.set_ylabel('Cantidad de missings')
ax.set_title('Missings en los que falta uno de los 2 datos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()
# -

# Se puede observar cómo, a excepción de la `presion_atmosferica` en el que tiene muy pocos missings, la cantidad de missings en la tarde suele ser más del doble que de temprano

# #### ¿Cómo podemos usar los datos de los horarios para rellenar los missings faltantes?

# Analizemos la relación que suelen tener los datos entre sí

columnas_con_horarios_opuestos = [
    'humedad',
    'nubosidad',
    'temperatura',
    'velocidad_viendo',
]

# +
fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, dpi=100, figsize=(15, 15))

i = j = 0
for columna in columnas_con_horarios_opuestos:
    col_tempr = columna + '_temprano'
    col_tarde = columna + '_tarde'
    axes[j, i].scatter(
        x=df[col_tempr], y=df[col_tarde], s=2,
    )
    axes[j, i].set_title(columna)
    axes[j, i].set_xlabel(col_tempr)
    axes[j, i].set_ylabel(col_tarde)
    i += 1
    if i == 2:
        j += 1
        i = 0

plt.show()
# -

# Podemos ver cómo los datos de la `nubosidad` sólo están cargados con números enteros en la mañana, y con decimales en la tarde. Por lo que no nos servirá mucho su distribución.

columnas_con_horarios_opuestos.remove('nubosidad')

# Sin embargo en los otros, podemos ver una clara tendencia de cada uno de sus valores

# Si calculamos el promedio que llevan la diferencia entre las columnas

relacion_horarios = {}
for columna in columnas_con_horarios_opuestos:
    col_tempr = columna + '_temprano'
    col_tarde = columna + '_tarde'
    relacion_horarios[col_tempr] = (
        df[df[col_tempr].notna()][col_tempr] - df[df[col_tempr].notna()][col_tarde]
    ).mean()
    relacion_horarios[col_tarde] = (
        df[df[col_tarde].notna()][col_tarde] - df[df[col_tarde].notna()][col_tempr]
    ).mean()


# Entonces con esto podemos rellenar los casos que se tenga un dato de los dos

# +
def llenar_nans_con_horario_alterno(fila, columna, columna_alt):
    if np.isnan(fila[columna]):
        if np.isnan(fila[columna_alt]):
            return np.nan
        return relacion_horarios[columna] + fila[columna_alt]
    return fila[columna]


for columna in columnas_con_horarios_opuestos:
    col_tempr = columna + '_temprano'
    col_tarde = columna + '_tarde'
    df[col_tempr] = df.apply(
        lambda fila: llenar_nans_con_horario_alterno(fila, col_tempr, col_tarde), axis=1
    )
    df[col_tarde] = df.apply(
        lambda fila: llenar_nans_con_horario_alterno(fila, col_tarde, col_tempr), axis=1
    )
# -

# De esta forma logramos reducir la cantidad de missings

# Desde la cantidad que taníamos inicialmente:

missings_totales_iniciales

# Hasta la actual:

df.isna().sum(axis=1).sum()

# ### Conversión de variables categóricas

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
    df[columna_viento + '_num'] = df.replace(
        puntos_del_compas.keys(), puntos_del_compas.values()
    )[columna_viento]
# -

# ## Análisis de los datos

# En esta seccion buscamos responder la pregunta: **Que relacion hay entre los features y el target?**
#
# Este análisis extensivo puede verse en [relaciones.py](relaciones.py), a continación se marcarán los puntos más importantes.

# ### Analisis Inicial

# En principio queremos saber : **Cual es la probabilidad de que lluevan hamburguesas?** para luego poder tomar ese valor de referencia.
#
# Aparte de eso, hacemos algunas transformaciones que seran de utilidad.

df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente'], inplace=True)

df['llovieron_hamburguesas_al_dia_siguiente_n'] = df.replace(
    {'llovieron_hamburguesas_al_dia_siguiente': {'si': 100, 'no': 0}}
)['llovieron_hamburguesas_al_dia_siguiente']

burger_mean = df['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
burger_mean


# ### Analisis de features numericas


def cmpNumeric(label):
    plt.figure(dpi=100)
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


def detNumeric(label, c=0, m=1, s=10, op=">"):
    x = []
    y = []
    for i in range(0, s):
        x.append(i * m + c)
        if op == ">":
            y.append(
                df[(df[label] > i * m + c)][
                    'llovieron_hamburguesas_al_dia_siguiente_n'
                ].mean()
            )
        else:
            y.append(
                df[(df[label] < i * m + c)][
                    'llovieron_hamburguesas_al_dia_siguiente_n'
                ].mean()
            )
    plt.suptitle(
        f"Probabilidad de que lluevan hamburguesas al dia siguiente segun {label}"
    )
    plt.plot(x, y)


# ---

cmpNumeric("horas_de_sol")


# Notamos que la tendencia en horas de sol es menor cuando lloveran hamburguesas al dia siguiente.
# Vemos los si se concentran por debajo de 7 y los no por arriba.

detNumeric("horas_de_sol", 7, -0.5, 20, "<")

# Notamos que dismunuye la posibilidad mientras mas horas de sol haya

# ---

cmpNumeric("humedad_tarde")

# Notamos que la tendencia en humedad tarde es mayor cuando lloveran hamburguesas al dia siguiente.
# Vemos los no se concentran por debajo de 60 y los si por arriba.

detNumeric("humedad_tarde", 60, 5, 20)

# Notamos que aumenta la posibilidad mientras mas humedad haya

# ---

cmpNumeric("nubosidad_tarde")

# Notamos que la tendencia en nubosidad tarde es mayor cuando lloveran hamburguesas al dia siguiente.
# Vemos los no se concentran por debajo de 7 y los si por arriba.

# ---

cmpNumeric("mm_lluvia_dia")

# Notamos que no suele llover previo a una lluvia de hamburguesas

detNumeric("mm_lluvia_dia", 0, 5, 15)

# Notamos que aumenta la posibilidad mientras mas mm lluvia haya

# ---

# #### Filtros en variables numericas

# Notamos ciertos puntos de inflexion en los graficos previos, a partir de ello nos preguntamos : **Cual es la probabilidad de que llueva pasado ese punto?**.

# +
df_num = pd.DataFrame(
    {
        'filtro': [
            "promedio normal",
            "horas_de_sol < 7",
            "humedad_tarde > 60",
            "nubosidad_tarde > 7",
            "mm_lluvia_dia > 5",
        ],
        '%_llovieron_hamburguesas': [
            burger_mean,
            df[(df['horas_de_sol'] < 7)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['humedad_tarde'] > 60)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['nubosidad_tarde'] > 7)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['mm_lluvia_dia'] > 5)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
        ],
    }
).sort_values(by="%_llovieron_hamburguesas", ascending=False)

plt.figure(dpi=150, figsize=(3, 2))
sns.barplot(data=df_num, y='filtro', x='%_llovieron_hamburguesas')
plt.axvline(burger_mean, 0, 1, color="#000")
plt.show()
df_num
# -

# Apreciamos que aumenta mucho la posibilidad de que lluevan hamburguesas

# +
df_num = pd.DataFrame(
    {
        'filtro': [
            "promedio normal",
            "horas_de_sol < 5",
            "humedad_tarde > 70",
            "nubosidad_tarde > 7.5",
            "mm_lluvia_dia > 10",
        ],
        '%_llovieron_hamburguesas': [
            burger_mean,
            df[(df['horas_de_sol'] < 5)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['humedad_tarde'] > 70)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['nubosidad_tarde'] > 7.5)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[(df['mm_lluvia_dia'] > 10)][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
        ],
    }
).sort_values(by="%_llovieron_hamburguesas", ascending=False)

plt.figure(dpi=150, figsize=(3, 2))
sns.barplot(data=df_num, y='filtro', x='%_llovieron_hamburguesas')
plt.axvline(burger_mean, 0, 1, color="#000")
plt.show()
df_num
# -

# Vemos que tocando los coeficientes podemos aumentar aun mas la posibilidad

# ### Analisis en features categoricas

labels_cat = ['llovieron_hamburguesas_hoy', 'barrio']


def cmpNonNumeric2(label):

    height = max(3, len(df[label].unique()) * 0.2)
    plt.figure(dpi=150, figsize=(5, height))

    plt.title("% de lluvia de haburguesas segun " + label)

    df2 = (
        df[[label, "llovieron_hamburguesas_al_dia_siguiente_n"]]
        .groupby(label)
        .mean()
        .sort_values("llovieron_hamburguesas_al_dia_siguiente_n")
    )
    sns.barplot(y=df2.index, x="llovieron_hamburguesas_al_dia_siguiente_n", data=df2)
    plt.axvline(burger_mean, 0, 1, color="#000")

    plt.show()


cmpNonNumeric2('llovieron_hamburguesas_hoy')

# Vemos una clara dependencia entre si llovieron hamburguesas un dia y si lloveran al siguiente

cmpNonNumeric2('barrio')

# Vemos que en algunos barrios llueve un poco mas que el pormedio y en otros considerablemente menos

# #### Filtros en variables numericas

# Notamos ciertos grupos con mayor probabilidad de lluvia en los graficos previos, a partir de ello nos preguntamos : **Cual es la probabilidad de que llueva perteneciendo a estos grupos?**.

# +
df_cat = pd.DataFrame(
    {
        'filtro': [
            "promedio normal",
            "llovieron_hamburguesas_hoy",
            "barrio in top 6",
            "barrio in top 3",
            "barrio in low 6",
            "barrio in low 3",
        ],
        '%_llovieron_hamburguesas': [
            burger_mean,
            df[(df["llovieron_hamburguesas_hoy"] == "si")][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
            df[
                (
                    df["barrio"].isin(
                        [
                            'Parque Patricios',
                            'Villa Pueyrredón',
                            'Saavedra',
                            'Chacarita',
                            'Recoleta',
                            'Vélez Sársfield',
                        ]
                    )
                )
            ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[
                (
                    df["barrio"].isin(
                        ['Parque Patricios', 'Villa Pueyrredón', 'Saavedra',]
                    )
                )
            ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[
                (
                    df["barrio"].isin(
                        [
                            'Villa Crespo',
                            'Palermo cheto',
                            'Villa Crespo',
                            'Villa Santa Rita',
                            'Parque Chacabuco',
                            'Balvanera',
                        ]
                    )
                )
            ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean(),
            df[(df["barrio"].isin(['Villa Crespo', 'Palermo cheto', 'Villa Crespo',]))][
                'llovieron_hamburguesas_al_dia_siguiente_n'
            ].mean(),
        ],
    }
).sort_values(by="%_llovieron_hamburguesas", ascending=False)

plt.figure(dpi=150, figsize=(3, 2))
sns.barplot(data=df_cat, y='filtro', x='%_llovieron_hamburguesas')
plt.axvline(burger_mean, 0, 1, color="#000")
plt.show()
df_cat


# -

# Notamos que que lluevan hamburguesas aumenta la posibilidad considerablemente, al mismo tiempo ciertos barrios la disminuyen considerablemente.

# ### Analisis Features Combinados

# #### Features numericas segun llovieron_hamburguesas_hoy


def detNumeric2(label, c=0, m=1, s=10, op=">"):
    plt.figure(dpi=100)
    x = []
    y = []
    z = []
    for i in range(0, s):
        x.append(i * m + c)
        if op == ">":
            y.append(
                df[
                    (df['llovieron_hamburguesas_hoy'] == 'si') & (df[label] > i * m + c)
                ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
            )
            z.append(
                df[
                    (df['llovieron_hamburguesas_hoy'] == 'no') & (df[label] > i * m + c)
                ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
            )
        else:
            y.append(
                df[
                    (df['llovieron_hamburguesas_hoy'] == 'si') & (df[label] < i * m + c)
                ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
            )
            z.append(
                df[
                    (df['llovieron_hamburguesas_hoy'] == 'no') & (df[label] < i * m + c)
                ]['llovieron_hamburguesas_al_dia_siguiente_n'].mean()
            )
    plt.suptitle(
        f"Probabilidad de que lluevan hamburguesas al dia siguiente segun {label}"
    )
    plt.plot(x, y)
    plt.plot(x, z)
    plt.legend(["llovieron hamburguesas hoy", "no llovieron hamburguesas hoy"])
    plt.show()


detNumeric2("humedad_tarde", 60, 5, 20)

# Notamos que como aumenta la posibilidad mientras mas humedad haya segun si llovieron hamburguesas

detNumeric2("horas_de_sol", 7, -0.5, 20, "<")

# Notamos que como disminuye la posibilidad mientras mas horas de sol haya segun si llovieron hamburguesas

# ## Conclusiones

# En este análisis exploratorio observamos cómo se comportaba cada variable individualmente con el target que queríamos predecir, en este caso si iba a llover hamburguesas al día siguiente. En este caso tratamos de hacer una inducción de la relación de las variables con si en cada caso cumplía con el target propuesto, para luego poder usarlas para el análisis y descartar el resto.
#
# Como resultado destacamos 1 condición categórica determinante (`llovieron_hamburguesas_hoy` sea que `si`) y otras condiciones 3 numéricas independientes entre sí, las cuales estarían directamente correlacionadas con la probabilidad de lluvia. Con estos datos, podemos construir una función baseline que haría una evaluación booleana, y de esta manera predice si al día siguiente va a llover o no.
#
# Ademas vemos diferentes probabilidades para algunas features numericas segun si `llovieron_hamburguesas_hoy`
#
# A continuación mostramos sus resultados.
#

# ## Baseline

# Planteamos una funcion baseline como condiciones booleanas basada en los filtros del analisis.


# +
def funcion_baseline(row):
    if row["llovieron_hamburguesas_hoy"] == "si":
        if row['horas_de_sol'] < 2:
            return True
        if row['nubosidad_tarde'] > 7:
            return True
        if row["humedad_tarde"] > 70:
            return True

    if row["mm_lluvia_dia"] > 10:
        return True
    if row["humedad_tarde"] > 80:
        return True

    return False


def baseline(df):
    return df.apply(funcion_baseline, axis=1)


# +

_baseline = baseline(df)
_target = df["llovieron_hamburguesas_al_dia_siguiente"] == "si"

(_baseline == _target).mean()
# -

# Para mejorar el rendimiento habria que mejorar el modelo.
#
# El modelo podria mejorar de las siguientes formas:
# - Agregando filtros negativos (que indiquen una reduccion en la posibilidad de que lluevan 🍔)
# - Agregando filtros de otras features
# - Agregando filtros intermedios
