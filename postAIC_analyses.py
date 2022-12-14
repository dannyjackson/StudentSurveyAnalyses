# cd Documents/AnBeh_Revised/SurveyResults
import pandas as pd
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statistics
from scipy.stats import sem
import math
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data_Nov22.csv')

# pre survey analyses
runs = pd.read_csv('AIC_results_pre.csv')
rows = ['Intercept', 'LGBTQ[T.LGBTQ]', 'Gender[T.Not_Man]',
       'Religion[T.Abrahamic]', 'Adj R-squared'
       ]
sig = pd.DataFrame(columns=[Qlist], index = rows)
coefficient = pd.DataFrame(columns=[Qlist], index = rows)
sig_all = pd.DataFrame(columns=[Qlist], index = rows)


rows_anova = ['pre_Q', 'LGBTQ', 'Gender', 'Religion',
       'Residual']
sig_anova = pd.DataFrame(columns=[Qlist], index = rows_anova)
coefficient_anova = pd.DataFrame(columns=[Qlist], index = rows_anova)



for index, row in runs.iterrows():
    Question = row['Question']
    LGBTQ = row['LGBTQ']
    Gender = row['Gender']
    Religion = row['Religion']
    stuff = f'{LGBTQ}', f'{Gender}', f'{Religion}',
    stuff = list(stuff)
    print(stuff)
    plus = " + "
    stuff2=list()
    for i in stuff.copy():
        if 'nan' not in i:
            stuff2.append(i)
    stuff2 = plus.join(stuff2)
    k = f'pre_{Question} ~ {stuff2}'
    print(k)
    model = smf.ols(formula=k, data=df).fit()
    model.summary()
    with open('pre_AIC_linear/linear/' + Question + '_summary.csv', 'w') as fh:
        fh.write(model.summary().as_csv())
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv('pre_AIC_linear/anova/' + Question + '_anova_summary.csv')
    model_summary = model.summary()
    model_as_html = model_summary.tables[1].as_html()
    df2 = pd.read_html(model_as_html, header=0, index_col=0)[0]
    s = df2['P>|t|'] < 0.05
    s = s.rename(index={f'pre_{Question}': 'pre_Q'})
    sig[Question] = s
    sig.loc['Adj R-squared', Question] = 1
    s_all = df2['P>|t|']
    s_all = s_all.rename(index={f'pre_{Question}': 'pre_Q'})
    sig_all[Question] = s_all
    t = model.rsquared_adj
    q = df2['coef']
    q = q.rename(index={f'pre_{Question}': 'pre_Q'})
    sa = anova_table['PR(>F)']
    sa = sa.rename(index={f'pre_{Question}': 'pre_Q'})
    sig_anova[Question] = sa
    ca = anova_table['sum_sq']
    ca = ca.rename(index={f'pre_{Question}': 'pre_Q'})
    coefficient_anova[Question] = ca
    coefficient[Question] = q
    coefficient.loc['Adj R-squared', Question] = t


sig = sig.astype(int)
sig_dir = coefficient * sig

sig_anova_t = sig_anova.transpose()
sig_anova_t.to_csv('pre_AIC_linear/anova/anova_sig.csv')
coefficient_anova_t = coefficient_anova.transpose()
coefficient_anova_t.to_csv('post_AIC_linear/anova/anova_coef.csv')

sig_dir_t = sig_dir.transpose()
sig_dir_t.to_csv('pre_AIC_linear/linear/linear_sig_direction.csv')

coefficient_t = coefficient.transpose()
coefficient_t.to_csv('pre_AIC_linear/linear/linear_coefficients.csv')
Then do Q28 on its own








# post survey analyses

runs = pd.read_csv('linear/AIC_results_noRace.csv')
runs = runs.drop(19)
runs = runs.drop(20)
d = dict()

Qlist = ['Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q58', 'Q37_total', 'Q38_total']


rows = ['Intercept', 'Course[T.Revised]', 'LGBTQ[T.LGBTQ]', 'Gender[T.Not_Man]',
       'Religion[T.Abrahamic]',
       'Course[T.Revised]:LGBTQ[T.LGBTQ]',
       'Course[T.Revised]:Gender[T.Not_Man]',
       'Course[T.Revised]:Religion[T.Abrahamic]',
       'pre_Q']
sig = pd.DataFrame(columns=[Qlist], index = rows)
coefficient = pd.DataFrame(columns=[Qlist], index = rows)
sig_all = pd.DataFrame(columns=[Qlist], index = rows)

rows_anova = ['Course', 'LGBTQ', 'Gender', 'Race', 'Religion', 'Course:LGBTQ',
       'Course:Gender', 'Course:Religion', 'Course:Race', 'pre_Q',
       'Residual']
sig_anova = pd.DataFrame(columns=[Qlist], index = rows_anova)
coefficient_anova = pd.DataFrame(columns=[Qlist], index = rows_anova)



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
    d[Question] = model.fittedvalues.values





# Then do Q28 and Q30 on their own
Question = 'Q28'
k = f'post_{Question} ~ pre_{Question}'
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



Question = 'Q30'
k = f'post_{Question} ~ pre_{Question}'
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



Q1 = df[["Course", "LGBTQ"]]
Q1['fit'] = d['Q1']

bx = sns.FacetGrid(Q11, col = "LGBTQ", margin_titles=True)
bx.set_titles(col_template="{col_name} ")
bx.set_axis_labels("LGBTQ")
bx.map(sns.boxplot, "fit", "Course", palette=['white'])
bx.map(sns.stripplot, "fit", "Course")
bx.savefig("interaction_boxplots/Q1.png")


Q2 = df[["Course", "Gender"]]
Q2['fit'] = d['Q2']

bx = sns.FacetGrid(Q2, col = "Gender", margin_titles=True)
bx.set_titles(col_template="{col_name} ")
bx.set_axis_labels("Gender")
bx.map(sns.boxplot, "fit", "Course", palette=['white'])
bx.map(sns.stripplot, "fit", "Course")
bx.savefig("interaction_boxplots/Q2.png")



df['post_Q3'].dropna()
Q3 = df.dropna(subset=['post_Q3', "pre_Q3"])
Q3 = Q3[["Course", "LGBTQ", "Gender"]]
Q3['fit'] = d['Q3']
d[Question] = model.fittedvalues.values

bx = sns.FacetGrid(Q3, row = "Gender", col = "LGBTQ", margin_titles=True)
bx.set_titles(col_template="{col_name} ", row_template="{row_name}")
bx.set_axis_labels("Gender", "LGBTQ")
bx.map(sns.boxplot, "fit", "Course", palette=['white'])
bx.map(sns.stripplot, "fit", "Course")
bx.savefig("interaction_boxplots/Q3.png")


Q11 = df[["Course", "LGBTQ", "Gender"]]
Q11['fit'] = d['Q11']

bx = sns.FacetGrid(Q11, row = "Gender", col = "LGBTQ", margin_titles=True)
bx.set_titles(col_template="{col_name} ", row_template="{row_name}")
bx.set_axis_labels("Gender", "LGBTQ")
bx.map(sns.boxplot, "fit", "Course", palette=['white'])
bx.map(sns.stripplot, "fit", "Course")
bx.savefig("interaction_boxplots/Q11.png")


# tukey LGBTQ pre vs post
Question = "Q1"
k = f'post_{Question} ~ pre_{Question} + Course + LGBTQ + Gender + Course:LGBTQ'
print(k)
model = smf.ols(formula=k, data=df).fit()
df_2 = df.loc[df['LGBTQ'] == '0_Not_LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q1'], groups=df_2['Course'], alpha=0.05))


# mean and SE for questions
# Q1
round(df.loc[df['LGBTQ'] == 'LGBTQ']['post_Q1'].dropna().mean(), 3)
# -0.906
round(df.loc[df['LGBTQ'] == 'LGBTQ']['post_Q1'].dropna().sem(), 3)
# 0.274
round(df.loc[df['LGBTQ'] == '0_Not_LGBTQ']['post_Q1'].dropna().mean(), 3)
# 0.507
round(df.loc[df['LGBTQ'] == '0_Not_LGBTQ']['post_Q1'].dropna().sem(), 3)
# 0.170



# Q2
# tukey LGBTQ pre vs post
df_2 = df.loc[df['LGBTQ'] == 'LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q2'], groups=df_2['Course'], alpha=0.05))

#   group1    group2 meandiff p-adj   lower  upper  reject
# 0_Unrevised Revised  -0.0714 0.8868 -1.0736 0.9308  False

df_2 = df.loc[df['LGBTQ'] == '0_Not_LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q2'], groups=df_2['Course'], alpha=0.05))
#   group1    group2 meandiff p-adj   lower  upper  reject
# 0_Unrevised Revised  -0.8217 0.0008 -1.2944 -0.349   True

df_2 = df.loc[df['Gender'] == 'Not_Man']
print(pairwise_tukeyhsd(endog=df_2['post_Q2'], groups=df_2['Course'], alpha=0.05))
#  group1    group2 meandiff p-adj   lower   upper  reject
# 0_Unrevised Revised  -0.5766 0.0211 -1.0653 -0.0878   True

df_2 = df.loc[df['Gender'] == '0_Man']
print(pairwise_tukeyhsd(endog=df_2['post_Q2'], groups=df_2['Course'], alpha=0.05))
#    group1    group2 meandiff p-adj   lower  upper  reject
# 0_Unrevised Revised  -0.7566 0.1759 -1.8664 0.3532  False


# Q3
#do Not LGBTQ revised vs Unrevised
df_2 = df.loc[df['LGBTQ'] == '0_Not_LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q3'], groups=df_2['Course'], alpha=0.05))
#   group1    group2 meandiff p-adj   lower  upper  reject
# 0_Unrevised Revised  -0.8217 0.0008 -1.2944 -0.349   True

# do LGBTQ revised vs Unrevised
df_2 = df.loc[df['LGBTQ'] == 'LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q3'], groups=df_2['Course'], alpha=0.05))
# group1    group2 meandiff p-adj   lower  upper  reject
# 0_Unrevised Revised  -0.3079 0.6012 -1.4835 0.8676  False

do Not Man revised vs unrevised
df_2 = df.loc[df['Gender'] == 'Not_Man']
print(pairwise_tukeyhsd(endog=df_2['post_Q3'], groups=df_2['Course'], alpha=0.05))

do Man revised vs unrevised
# tukey LGBTQ pre vs post
df_2 = df.loc[df['LGBTQ'] == 'LGBTQ']
print(pairwise_tukeyhsd(endog=df_2['post_Q2'], groups=df_2['Course'], alpha=0.05))









# mean and SE for questions
# Q1
round(df.loc[df['LGBTQ'] == 'LGBTQ']['post_Q1'].dropna().mean(), 3)
# -0.906
round(df.loc[df['LGBTQ'] == 'LGBTQ']['post_Q1'].dropna().sem(), 3)
# 0.274
round(df.loc[df['LGBTQ'] == '0_Not_LGBTQ']['post_Q1'].dropna().mean(), 3)
# 0.507
round(df.loc[df['LGBTQ'] == '0_Not_LGBTQ']['post_Q1'].dropna().sem(), 3)
# 0.170
