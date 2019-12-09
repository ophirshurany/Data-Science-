# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:51:46 2019

@author: OphirShurany
"""
#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
#%% create dataframe
df = pd.read_csv("bank.csv",sep='|',encoding='utf8')
#view first 5 rows in df
df.head()
#presenting all columns, number of rows and type
info=df.info()
#feature statistics for numerical categories
stats=df.describe()
#change "yes" or "no" to 1 or 0
df['y'] = df.y.eq('yes').mul(1)
dic={v: k for k,v in enumerate(calendar.month_abbr)}
###לא הבנתי את השאלה של הרבעונים

#%%Missing Values
df.isnull().sum()