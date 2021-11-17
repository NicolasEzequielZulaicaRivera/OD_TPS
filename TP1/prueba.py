# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

features = pd.read_csv("datasets/features.csv", low_memory=False)
target = pd.read_csv("datasets/target.csv")
df = features.merge(target, on='id')
df['presion_atmosferica_tarde'] = pd.to_numeric(
    df['presion_atmosferica_tarde'], errors='coerce'
)
df.dropna(subset=["llovieron_hamburguesas_al_dia_siguiente"], inplace=True)
df.info()

df[
    (df["horas_de_sol"] < 7)
    & (df["humedad_tarde"] > 60)
    & (df["mm_lluvia_dia"].isna())
    & (df["nubosidad_tarde"] > 7)
]["llovieron_hamburguesas_al_dia_siguiente"].value_counts()

df[(df["horas_de_sol"] < 7) & (df["humedad_tarde"] > 60)][
    "llovieron_hamburguesas_al_dia_siguiente"
].value_counts()

df[
    (df["direccion_viento_tarde"].isin(['Noroeste', 'Oestenoroeste', 'Norte', 'Oeste']))
    & (
        df["rafaga_viento_max_direccion"].isin(
            [
                'Noroeste',
                'Oestenoroeste',
                'Norte',
                'Oeste',
                'Nornoreste',
                'Oestesuroeste',
            ]
        )
    )
    & (
        df["barrio"].isin(
            [
                'Parque Patricios',
                'Villa Pueyrredón',
                'Saavedra',
                'Chacarita',
                'Recoleta',
                'Vélez Sársfield',
                'Barracas',
                'Villa del Parque',
                'Villa Devoto',
                'Caballito',
                'Monte Castro',
                'La Boca',
                'Villa Soldati',
                'San Cristóbal',
                'Monserrat',
                'Flores',
                'Parque Avellaneda',
                'Constitución',
                'Puerto Madero',
                'Boedo',
                'Villa Real',
                'La Paternal',
                'Villa Riachuelo',
            ]
        )
    )
    & (df["llovieron_hamburguesas_hoy"] == "si")
]["llovieron_hamburguesas_al_dia_siguiente"].value_counts()

df[(df["mm_lluvia_dia"].isna()) & (df["llovieron_hamburguesas_hoy"] == "si")]

filtrosFuertes = [df["mm_lluvia_dia"].isna(), df["llovieron_hamburguesas_hoy"] == "si"]
filtrosDebiles = [
    df["horas_de_sol"] < 7,
    df["humedad_tarde"] > 60,
    df["nubosidad_tarde"] > 7,
    df["rafaga_viento_max_direccion"].isin(
        ['Noroeste', 'Oestenoroeste', 'Norte', 'Oeste', 'Nornoreste', 'Oestesuroeste']
    ),
    df["barrio"].isin(
        [
            'Parque Patricios',
            'Villa Pueyrredón',
            'Saavedra',
            'Chacarita',
            'Recoleta',
            'Vélez Sársfield',
            'Barracas',
            'Villa del Parque',
            'Villa Devoto',
            'Caballito',
            'Monte Castro',
            'La Boca',
            'Villa Soldati',
            'San Cristóbal',
            'Monserrat',
            'Flores',
            'Parque Avellaneda',
            'Constitución',
            'Puerto Madero',
            'Boedo',
            'Villa Real',
            'La Paternal',
            'Villa Riachuelo',
        ]
    ),
]

df[
    (df["horas_de_sol"] < 7)
    & (df["humedad_tarde"] > 60)
    & (df["nubosidad_tarde"] > 7)
    & (
        df["direccion_viento_tarde"].isin(
            ['Noroeste', 'Oestenoroeste', 'Norte', 'Oeste']
        )
    )
    & (
        df["rafaga_viento_max_direccion"].isin(
            [
                'Noroeste',
                'Oestenoroeste',
                'Norte',
                'Oeste',
                'Nornoreste',
                'Oestesuroeste',
            ]
        )
    )
    & (
        df["barrio"].isin(
            [
                'Parque Patricios',
                'Villa Pueyrredón',
                'Saavedra',
                'Chacarita',
                'Recoleta',
                'Vélez Sársfield',
                'Barracas',
                'Villa del Parque',
                'Villa Devoto',
                'Caballito',
                'Monte Castro',
                'La Boca',
                'Villa Soldati',
                'San Cristóbal',
                'Monserrat',
                'Flores',
                'Parque Avellaneda',
                'Constitución',
                'Puerto Madero',
                'Boedo',
                'Villa Real',
                'La Paternal',
                'Villa Riachuelo',
            ]
        )
    )
]["llovieron_hamburguesas_al_dia_siguiente"].value_counts()


# +
def baseline(df):
    c = (
        0.5 * (df["mm_lluvia_dia"].isna())
        + 0.5 * (df["llovieron_hamburguesas_hoy"] == "si")
        + 0.3 * (df["horas_de_sol"] < 7)
        + 0.3 * (df["humedad_tarde"] > 60)
        + 0.3 * (df["nubosidad_tarde"] > 7)
        + 0.1
        * (
            df["rafaga_viento_max_direccion"].isin(
                [
                    'Noroeste',
                    'Oestenoroeste',
                    'Norte',
                    'Oeste',
                    'Nornoreste',
                    'Oestesuroeste',
                ]
            )
        )
        + 0.1
        * (
            df["barrio"].isin(
                [
                    'Parque Patricios',
                    'Villa Pueyrredón',
                    'Saavedra',
                    'Chacarita',
                    'Recoleta',
                    'Vélez Sársfield',
                    'Barracas',
                    'Villa del Parque',
                    'Villa Devoto',
                    'Caballito',
                    'Monte Castro',
                    'La Boca',
                    'Villa Soldati',
                    'San Cristóbal',
                    'Monserrat',
                    'Flores',
                    'Parque Avellaneda',
                    'Constitución',
                    'Puerto Madero',
                    'Boedo',
                    'Villa Real',
                    'La Paternal',
                    'Villa Riachuelo',
                ]
            )
        )
    )

    return c > 1


_baseline = baseline(df)
_target = df["llovieron_hamburguesas_al_dia_siguiente"] == "si"

_res=(_baseline == _target)
_notres=(_baseline != _target)


# +
def baseline2(df, threshold=1, vals=[0.49, 0.49, 0.46, 0.45, 0.44, 0.31]):
    c = (
        vals[0] * (df["nubosidad_tarde"] > 7)
        + vals[1] * (df["mm_lluvia_dia"].isna())
        + vals[2] * (df["llovieron_hamburguesas_hoy"] == "si")
        + vals[3] * (df["humedad_tarde"] > 60)
        + vals[4] * (df["horas_de_sol"] < 7)
        + vals[5] * (df["barrio"].isin(['Parque Patricios','Villa Pueyrredón','Saavedra','Chacarita','Recoleta','Vélez Sársfield','Barracas','Villa del Parque','Villa Devoto',]))
        
        - .2 * (df["llovieron_hamburguesas_hoy"] != "si")
        - .2 * (df["barrio"].isin(["Estesureste","Este","Estenoreste","Noreste","Sureste"]))
        - .2 * (df["nubosidad_tarde"] < 4)
        - .2 * (df["humedad_tarde"] < 30)
        - .2 * (df["mm_lluvia_dia"].isna()==False)
    )

    return c > threshold

_baseline = baseline2(df)
_target = df["llovieron_hamburguesas_al_dia_siguiente"] == "si"

_res=(_baseline == _target)
_notres=(_baseline != _target)

_res.mean()
# -



# +
vc = df["llovieron_hamburguesas_al_dia_siguiente"].value_counts()
t = df[ _res ]["llovieron_hamburguesas_al_dia_siguiente"].value_counts()
f = df[ _notres ]["llovieron_hamburguesas_al_dia_siguiente"].value_counts()

print("TP:"+str( t['si'] )+"\tFP:"+str( f['si'] ))
print("FN:"+str( f['no'] )+"\tTN:"+str( t['no'] ))
