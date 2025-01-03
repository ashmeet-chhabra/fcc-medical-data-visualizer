import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
# df = None
df = pd.read_csv('medical_examination.csv')

# 2
# df['overweight'] = None
bmi = df['weight'] * 10_000 / (df['height']**2)
df['overweight'] = np.where(bmi > 25, 1, 0)


# 3

df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1

df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])
df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
# print(df_cat)
sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_cat, kind='bar')
plt.show()


# 4
def draw_cat_plot():
    # 5
    # df_cat = None
    df_cat = pd.melt(df, value_vars=['cholestrol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])


    # 6
    # df_cat = None
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholestrol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size()
    

    # 7
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_cat, kind='bar')
    plt.show()

    # 8
    fig = None


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
