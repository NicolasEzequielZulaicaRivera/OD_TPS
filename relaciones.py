# -*- coding: utf-8 -*-
# ## Imports

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

# #### Dataset

features = pd.read_csv("datasets/features.csv", low_memory=False)

target = pd.read_csv("datasets/target.csv")

df = features.merge(target, on='id')
df['presion_atmosferica_tarde'] = pd.to_numeric(
    df['presion_atmosferica_tarde'], errors='coerce'
)

df.replace({'llovieron_hamburguesas_al_dia_siguiente': {'si': 100, 'no': 0}},inplace=True)

# ## Relaciones

# ### Analisis Exploratorio

# #### Cual es el porcentaje de lluvia de hamburguesas ?

burger_mean = df['llovieron_hamburguesas_al_dia_siguiente'].mean()
burger_mean


# #### Features Numericas

# ##### Analisis 1

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


labels = df.select_dtypes(include=np.number).columns.tolist()
for label in labels:
    cmpNumeric(label)

# ##### Observaciones 1

# Se nota que algunas features tienen tendencias mas marcadas que otras.
#
# Estas son las que parecieran mas relevantes y sus puntos de inflexion.
#
# * Horas de sol: 7 (si / no)
# * Humedad tarde: 60 (no /si)
# * mm lluvia: 1 (no / si)
# * Nubosidad tarde: 7 (no / si)

# ##### Analisis 2

labels = ["horas_de_sol","humedad_tarde","mm_lluvia_dia","nubosidad_tarde"]
means = []
for label in labels :
    means.append( df[[label,"llovieron_hamburguesas_al_dia_siguiente"]][ df[label].isna() ].mean()['llovieron_hamburguesas_al_dia_siguiente'] )
df_missings = pd.DataFrame(
    {'Label': labels, 'llovieron_hamburguesas_al_dia_siguiente': means}
).sort_values(by="llovieron_hamburguesas_al_dia_siguiente", ascending=False)

# +

plt.figure()

plt.title("% de lluvia de haburguesas cuando 'Label' es Nan")
sns.barplot( y="Label", x="llovieron_hamburguesas_al_dia_siguiente", data=df_missings )
plt.axvline(burger_mean, 0, 1)

plt.show()
# -

for label in labels:
    plt.figure()
    plt.title("")
    sns.histplot( data=df, x=label, bins=10, hue="llovieron_hamburguesas_al_dia_siguiente" )
    plt.show()

# Que el valor sea Nan solo parece aportar info relevante para mm_lluvia_dia

# ##### Observaciones 2

# TODO

# #### Features Categoricas

labels = df.select_dtypes(exclude=np.number).columns.tolist()
labels.remove('dia')


# ##### Analisis 1

def cmpNonNumeric(label):
    df2 = (
        df[[label, 'llovieron_hamburguesas_al_dia_siguiente']]
        .groupby(label)['llovieron_hamburguesas_al_dia_siguiente']
        .value_counts()
        .unstack()
    )
    df2.plot(kind='bar', stacked=True)


for label in labels:
    cmpNonNumeric(label)


# ##### Observaciones 1

# Estos graficos son horribles
#
# Se ve que ( llovieron_hamburguesas_al_dia_siguiente == si ) es mas probable si  ( llovieron_hamburguesas_hoy  == si )

# ##### Analisis 2

def cmpNonNumeric2(label):
    
    height = max( 3, len(df[label].unique()) *0.2 )
    plt.figure(dpi=150, figsize=(5, height) )
    
    plt.title("% de lluvia de haburguesas segun "+label)

    df2 = df[ [label,"llovieron_hamburguesas_al_dia_siguiente"] ].groupby(label).mean().sort_values("llovieron_hamburguesas_al_dia_siguiente")
    sns.barplot(y=df2.index, x="llovieron_hamburguesas_al_dia_siguiente", data=df2)
    plt.axvline(burger_mean, 0, 1)

    plt.show()


for label in labels:
    cmpNonNumeric2(label)

# ##### Observaciones 2

# Todas parecieran ponderar, entre una opcion y otra puede haber el doble de % de casos de lluvia de 🍔
#
# De especial utilidad llovieron_hamburguesas_hoy, que es simple y se nota que el triple del % cuando es 'si' que cuando es 'no'

# #### Feature Dia

# ##### Analisis

df['y'] = df['dia'].map(lambda x: x.split('-')[0])
df['m'] = df['dia'].map(lambda x: x.split('-')[1])
df['d'] = df['dia'].map(lambda x: x.split('-')[2])

plt.figure(dpi=150, figsize=(5, 3))
plt.title("Year")
sns.barplot( data=df, y='y', x='llovieron_hamburguesas_al_dia_siguiente', order=sorted(df['y'].unique()) )
plt.axvline(burger_mean, 0, 1)
plt.show()

plt.figure(dpi=150, figsize=(5, 3))
plt.title("Month")
sns.barplot( data=df, y='m', x='llovieron_hamburguesas_al_dia_siguiente', order=sorted(df['m'].unique()) )
plt.axvline(burger_mean, 0, 1)
plt.show()

plt.figure(dpi=150, figsize=(5, 5))
plt.title("Day")
sns.barplot( data=df, y='d', x='llovieron_hamburguesas_al_dia_siguiente', order=sorted(df['d'].unique()) )
plt.axvline(burger_mean, 0, 1)
plt.show()

# ##### Observaciones
