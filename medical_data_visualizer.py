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
    
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 
'overweight'])
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']


    # # 7
    # df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
 
    # 8
    fig = sns.catplot(x='variable', hue='value', data=df_cat, kind='count', col='cardio', order=cat_order).set_ylabels('total').fig
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # df_heat = None
    cholesterol_mask = df['ap_lo'] <= df['ap_hi']
    height_mask = (df['height'] <= df['height'].quantile(0.975)) & (df['height'] >= df['height'].quantile(0.025))
    weight_mask = (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))
    df_heat = df[cholesterol_mask & height_mask & weight_mask]


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
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f')
    # sns.heatmap(corr, mask=mask, annot=True)


    # 16
    fig.savefig('heatmap.png')
    return fig
