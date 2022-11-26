import pandas as pd
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data_Nov22.csv')

runs = pd.read_csv('linear/AIC_results_noRace.csv')
runs = runs.drop(19)
runs = runs.drop(20)



rows = ['Intercept', 'Course[T.Revised]', 'LGBTQ[T.LGBTQ]', 'Gender[T.Not_Man]',
       'Religion[T.Abrahamic]',
       'Course[T.Revised]:LGBTQ[T.LGBTQ]',
       'Course[T.Revised]:Gender[T.Not_Man]',
       'Course[T.Revised]:Religion[T.Abrahamic]',
       'pre_Q']
sig = pd.DataFrame(columns=[list], index = rows)
coefficient = pd.DataFrame(columns=[list], index = rows)
sig_all = pd.DataFrame(columns=[list], index = rows)

rows_anova = ['Course', 'LGBTQ', 'Gender', 'Race', 'Religion', 'Course:LGBTQ',
       'Course:Gender', 'Course:Religion', 'Course:Race', 'pre_Q',
       'Residual']
sig_anova = pd.DataFrame(columns=[list], index = rows_anova)
coefficient_anova = pd.DataFrame(columns=[list], index = rows_anova)



for index, row in runs.iterrows():
    Question = row['Question']
    Course = row['Course']
    LGBTQ = row['LGBTQ']
    Gender = row['Gender']
    Religion = row['Religion']
    Course_LGBTQ = row['Course:LGBTQ']
    Course_Gender = row['Course:Gender']
    Course_Religion = row['Course:Religion']
    stuff = f'{Course}', f'{LGBTQ}', f'{Gender}', f'{Religion}', f'{Course_LGBTQ}', f'{Course_Gender}', f'{Course_Religion}',
    stuff = list(stuff)
    print(stuff)
    plus = " + "
    stuff2=list()
    for i in stuff.copy():
        if 'nan' not in i:
            stuff2.append(i)
    stuff2 = plus.join(stuff2)
    k = f'post_{Question} ~ pre_{Question} + {stuff2}'
    print(k)
    model = smf.ols(formula=k, data=df).fit()
    model.summary()
    with open('post_AIC_linear/linear/' + Question + '_summary.csv', 'w') as fh:
        fh.write(model.summary().as_csv())
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv('post_AIC_linear/anova/' + Question + '_anova_summary.csv')
    model_summary = model.summary()
    model_as_html = model_summary.tables[1].as_html()
    df2 = pd.read_html(model_as_html, header=0, index_col=0)[0]
    s = df2['P>|t|'] < 0.05
    s = s.rename(index={f'pre_{Question}': 'pre_Q'})
    sig[Question] = s
    sig.loc['Intercept_all', Question] = 1
    s_all = df2['P>|t|']
    s_all = s_all.rename(index={f'pre_{Question}': 'pre_Q'})
    sig_all[Question] = s_all
    t = df2['coef']['Intercept']
    q = df2['coef']
    q = q.rename(index={f'pre_{Question}': 'pre_Q'})
    sa = anova_table['PR(>F)']
    sa = sa.rename(index={f'pre_{Question}': 'pre_Q'})
    sig_anova[Question] = sa
    ca = anova_table['sum_sq']
    ca = ca.rename(index={f'pre_{Question}': 'pre_Q'})
    coefficient_anova[Question] = ca
    coefficient[Question] = q
    coefficient.loc['Intercept_all', Question] = t

sig = sig.astype(int)
sig_dir = coefficient * sig

sig_anova_t = sig_anova.transpose()
sig_anova_t.to_csv('post_AIC_linear/anova/anova_sig.csv')
coefficient_anova_t = coefficient_anova.transpose()
coefficient_anova_t.to_csv('post_AIC_linear/anova/anova_coef.csv')

sig_dir_t = sig_dir.transpose()
sig_dir_t.to_csv('post_AIC_linear/linear/linear_sig_direction.csv')

coefficient_t = coefficient.transpose()
coefficient_t.to_csv('post_AIC_linear/linear/linear_coefficients.csv')
Then do Q28 on its own
