# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

features = pd.read_csv("datasets/features.csv", low_memory=False)

target = pd.read_csv("datasets/target.csv")

df = features.merge(target, on='id')
df['presion_atmosferica_tarde'] = pd.to_numeric(
    df['presion_atmosferica_tarde'], errors='coerce'
)


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


def cmpNonNumeric(label):
    df2 = df[[label,'llovieron_hamburguesas_al_dia_siguiente']].groupby(label)['llovieron_hamburguesas_al_dia_siguiente'].value_counts().unstack()
    df2.plot(kind='bar', stacked=True)


labels = df.select_dtypes(exclude=np.number).columns.tolist()
labels.remove('dia')
labels.remove('llovieron_hamburguesas_al_dia_siguiente')

for label in labels:
    cmpNonNumeric(label)
