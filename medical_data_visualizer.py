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


# 4
def draw_cat_plot():
    # 5
    # df_cat = None
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])


    # 6
    # df_cat = None
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size()
    

    # 7
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    plt.show()

    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # df_heat = None
    df_heat = df[df['ap_lo'] <= df['ap_hi']]
    df_heat = df_heat[df_heat['height'] <= df_heat['height'].quantile(0.9725)]
    df_heat = df_heat[df_heat['weight'] >= df_heat['weight'].quantile(0.025)]
    df_heat = df_heat[df_heat['weight'] <= df_heat['weight'].quantile(0.9725)]
    df_heat = df_heat[df_heat['height'] >= df_heat['height'].quantile(0.025)]


    # 12
    # corr = None
    corr = df_heat.corr()

    # 13
    # mask = None
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    # fig, ax = None
    fig, ax = plt.subplots(figsize=(12, 6))

    # 15
    sns.heatmap(corr, mask=mask)


    # 16
    fig.savefig('heatmap.png')
    return fig
