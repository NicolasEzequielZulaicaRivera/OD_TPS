# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("datasets/features.csv")

df2 = pd.read_csv("datasets/target.csv")

df = pd.merge(df, df2, on='id', how='outer')

from pandas_profiling import ProfileReport

report = ProfileReport(df, title='Llovieron hamburguesas', explorative=True, lazy=False)

report
