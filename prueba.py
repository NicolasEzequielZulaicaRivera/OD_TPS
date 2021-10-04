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

(_baseline == _target).mean()
