# Data analysis

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('cleanData_danny.csv')



# Modeling with interactions
# in python

import statsmodels.formula.api as smf



list = ['Q16', 'Q21', 'Q1', 'Q3', 'Q58', 'Q9', 'Q7', 'Q8', 'Q10', 'Q11', 'Q13', 'Q14', 'Q17', 'Q18', 'Q19', 'Q2', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']

sig_2 = pd.DataFrame(columns=[list], index = df4.index)
direction_2 = pd.DataFrame(columns=[list], index = df4.index)

q.index = direction_2.index
# model = smf.ols(formula=f'diff_{i} ~ Course + LGBTQ + Gender + Race + AnxietyDepression + Religion + Course*LGBTQ + Course*Gender + Course*Religion  + Course*AnxietyDepression', data=df).fit()

for i in list:
    model = smf.ols(formula=f'diff_{i} ~ Course_Revised + LGBTQ_LGBTQ + Gender_Not_Man + Race_PEER + Religion_Christian + Course_Revised*LGBTQ_LGBTQ + Course_Revised*Gender_Not_Man + Course_Revised*Religion_Christian + Course_Revised*Race_PEER', data=df2).fit()
    model.summary()
    with open('linear/' + i + '_summary.csv', 'w') as fh:
        fh.write(model.summary().as_csv())

    model_summary = model.summary()
    model_as_html = model_summary.tables[1].as_html()
    df4 = pd.read_html(model_as_html, header=0, index_col=0)[0]
    s = df4['P>|t|'] < 0.05
    sig_2[i] = s
    q = df4['coef'] < 0
    direction_2[i] = q

direction_2 = direction_2.astype(int)
direction_2 = direction_2 * -1
direction_2 = direction_2.replace(0,1)
sig_2 = sig_2.astype(int)
sig_dir = direction_2 * sig_2

sig_dir_t = sig_dir.transpose()
sig_dir_t.to_csv('linear/linear_sig_direction.csv')







for i in list:
    df3 = pd.read_csv('log/' + i + '_log_summary.csv', skiprows=9)
    df3 = df3.iloc[:9 , :]
    s = df3['P>|z| '] < 0.05
    s.index = sig.index
    sig[i] = s

sig = sig.astype(int)
for i in list:
    df4 = pd.read_csv('linear/' + i + '_summary.csv', skiprows=9)
    df3 = df3.iloc[:9 , :]
    s = df3['P>|z| '] < 0.05
    s.index = sig.index
    sig[i] = s

df4 = pd.read_csv('linear/' + i + '_summary.csv', skiprows=10)



df_5 = pd.read_csv('tokelsey.csv')







####################################################################################################
# Unused notes below
####################################################################################################





















# pd.CategoricalDtype(categories=['Strongly disagree', 'Disagree', 'Slightly disagree', 'I am unsure', 'Slightly agree', 'Agree', 'Strongly agree'], ordered=True)

pd.CategoricalDtype(categories=['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'], ordered=True)

df['Course'] = df.loc[:, 'pre_Course']

df['LGBTQ'].value_counts().to_csv('LGBTQ_Counts.csv')
df['Gender'].value_counts().to_csv('Gender_Counts.csv')
df['Religion'].value_counts().to_csv('Religion_Counts.csv')
df['Race'].value_counts().to_csv('Race_Counts.csv')
df['Course'].value_counts().to_csv('Course_Counts.csv')
df[['LGBTQ', 'Gender']].value_counts()
df[['Race', 'Gender']].value_counts()

cattype = pd.CategoricalDtype(categories=[-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2., 3.0, 4.0, 5.0, 6.0], ordered=True)



list = ['Q16', 'Q21', 'Q1', 'Q3', 'Q58', 'Q9', 'Q7', 'Q8', 'Q10', 'Q11', 'Q13', 'Q14', 'Q17', 'Q18', 'Q19', 'Q2', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']


df2 = pd.get_dummies(df, columns=['Course', 'LGBTQ', 'Gender', 'Religion', 'Race'])

pd.CategoricalDtype(categories=['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'], ordered=True)

for i in list:

df2 = df2[df2['diff_' + i].notna()]
df2['Course'] = df2.loc[:, 'pre_Course']
df2['diff_' + i] = df2['diff_' + i].astype(cattype)

mod_prob = OrderedModel(df2['diff_' + i],
                        df2[['Course_Revised', 'LGBTQ_LGBTQ', 'Gender_Not_Man', 'Religion_Christian', 'Race_PEER']],
                        distr='probit')

res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()



mod_log = OrderedModel(df2['diff_' + i], df2[['Course_Revised', 'LGBTQ_LGBTQ', 'Gender_Not_Man', 'Religion_Christian', 'Race_PEER']], distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()
with open('log/' + i + '_log_summary.csv', 'w') as fh:
    fh.write(res_log.summary().as_csv())




num_of_thresholds = 12
mod_prob.transform_threshold_params(res_prob.params[-num_of_thresholds:])

sig = pd.DataFrame(columns=[list], index = ['Course', 'LGBTQ', 'Gender', 'Religion_Christian', 'Religion_Atheist', 'Religion_Muslim', 'Race_Asian', 'Race_PEER', 'AnxietyDepression'])



for i in list:
    df3 = pd.read_csv('log/' + i + '_log_summary.csv', skiprows=9)
    df3 = df3.iloc[:9 , :]
    s = df3['P>|z| '] < 0.05
    s.index = sig.index
    sig[i] = s

sig = sig.astype(int)

sig.to_csv('log/significance.csv')

direction = pd.DataFrame(columns=[list], index = ['Course', 'LGBTQ', 'Gender', 'Religion_Christian', 'Religion_Atheist', 'Religion_Muslim', 'Race_Asian', 'Race_PEER', 'AnxietyDepression'])

for i in list:
    df3 = pd.read_csv('log/' + i + '_log_summary.csv', skiprows=9)
    df3 = df3.iloc[:9 , :]
    s = df3['   coef   '] < 0
    s.index = direction.index
    direction[i] = s

direction = direction.astype(int)
direction = direction * -1
direction = direction.replace(0,1)
direction = direction * sig


direction.to_csv('log/sig_direction.csv')





# in R
df.to_csv('dataframe_toR.csv')

model <- lm(Course ~ LGBTQ + Gender + post_QID40 + AnxietyDepression + pre_Q52 + Course*LGBTQ + Course*Gender + Course*pre_Q52  + Course*AnxietyDepression, data = x)
t <- summary(model)
write.csv(t, "model.csv")


lm(Course ~ LGBTQ, data = x)

y_true = x1+x2*0.5+x3*2+x4+ x1*x2 -x3*x2 + x4*x2


import statsmodels.api as sm
Xb = sm.add_constant(df[['diff_' + i, 'Course', 'LGBTQ', 'Gender', 'Q52', '', 'Religion_Muslim', 'Race_Asian', 'Race_PEER', 'AnxietyDepression']])
mod = sm.OLS(y_true, Xb)
res = mod.fit()
res.summary()
