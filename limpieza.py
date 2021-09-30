# -*- coding: utf-8 -*-
# # Limpieza de datos

# ## Carga

# Empezamos importando las librerías y cargando los datasets en memoria

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

df = pd.read_csv("datasets/features.csv", low_memory=False)

df2 = pd.read_csv("datasets/target.csv")

# Mezclamos los dos datasets para tener más comodidad

df = pd.merge(df, df2, on='id', how='outer')

# ## Verificando la calidad de los datos

# Queremos usar `NaN` como el valor nulo, para esto debemos uniformizar si hay otro tipo de dato faltantes

# Chequeamos que no haya celdas con `-` o espacios, usando los regexs `^-*[0-9]*\$` y `^ *\$`

regex1 = re.compile("^-*[0-9]*\$")
regex2 = re.compile("^ *\$")
df.astype(str).applymap(lambda x: bool(regex1.match(x) or regex2.match(x))).any(0)

# Vemos cuáles son las columnas que son numéricas

df[df.select_dtypes(include=[np.number]).columns]

# Exceptuando las relacionadas con la temperatura, ninguna de las columnas debería tener valores negativos

columnas_a_verificar = df.select_dtypes(include=[np.number]).columns.drop(
    ['temp_min', 'temp_max', 'temperatura_tarde', 'temperatura_temprano']
)
(df[columnas_a_verificar] < 0).any().to_frame('Menor que 0 ?')

# Tambien verificamos que no haya filas duplicadas

len(df[df.duplicated(keep=False)])
