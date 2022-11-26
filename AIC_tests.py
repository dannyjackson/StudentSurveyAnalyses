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

list = ['Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q58', 'Q37_total', 'Q38_total']

df = pd.read_csv('data_Nov22.csv')
# Question 1

model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Race + Religion + Course:LGBTQ + Course:Gender + Course:Religion + Course:Race', data=df).fit()

y, X = patsy.dmatrices(f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Race + Religion + Course:LGBTQ + Course:Gender + Course:Religion + Course:Race', data=df)

X_df.as.data.frame(X)

backwardSelection(X, y)



# Question 38
model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Race + Religion + Course:LGBTQ + Course:Gender + Course:Religion + Course:Race', data=df).fit()
1147.1038637584604

model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Race + Religion + Course:LGBTQ + Course:Religion + Course:Race', data=df).fit()
model.aic
1145.1064299747072

model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Race + Religion + Course:LGBTQ + Course:Religion', data=df).fit()
model.aic
1143.3543189984177
model.summary()


model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Gender + Religion + Course:LGBTQ + Course:Religion', data=df).fit()
1141.5616645102207

model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Religion + Course:LGBTQ + Course:Religion', data=df).fit()
1139.8036014865088

model1 = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + LGBTQ + Religion + Course:LGBTQ', data=df).fit()
model.aic
1138.3862467611125
model.summary()

model2 = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + Religion + Course:LGBTQ', data=df).fit()
1138.3862467611125
model.summary()


model = smf.ols(formula=f'post_{i} ~ pre_{i} + Course + Religion', data=df).fit()
model.summary()
1139.4545660663748
