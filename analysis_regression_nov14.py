# cd Documents/AnBeh_Revised/SurveyResults
# Relevant websites
# https://joelcarlson.github.io/2016/05/10/Exploring-Interactions/
# https://towardsdatascience.com/multiple-linear-regression-with-interactions-unveiled-by-genetic-programming-4cc325ac1b65
# https://www.restore.ac.uk/srme/www/fac/soc/wie/research-new/srme/modules/mod5/7/index.html
# https://www.nickmccullum.com/python-machine-learning/logistic-regression-python/
# https://statisticsbyjim.com/regression/choosing-regression-analysis/
# https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31
# https://www.statsmodels.org/dev/examples/notebooks/generated/ordinal_regression.html#Logit-ordinal-regression:
# Unveiling Concealable Stigmatized Identities in Class: The Impact of an Instructor Revealing Her LGBTQ+ Identity to Students in a Large-Enrollment Biology Course
# https://www.lifescied.org/doi/pdf/10.1187/cbe.21-06-0162

import pandas as pd
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


# df = pd.read_csv('SurveyData_12AUG2022.csv')

df = pd.read_csv('SurveyData_12AUG2022.csv')

# df = df.iloc[1: , :]

df.Q54.value_counts()

df['LGBTQ'] = df.loc[:, 'Q54']

df.loc[df["LGBTQ"] == "Yes", "LGBTQ"] = 'LGBTQ'
df.loc[df["LGBTQ"] == "No", "LGBTQ"] = 'Not_LGBTQ'
df.loc[df["LGBTQ"] == "Decline to state", "LGBTQ"] = 'Not_LGBTQ'

df.LGBTQ.value_counts()

df.Q40.value_counts()
df['Gender'] = df.loc[:, 'Q40']

df.loc[df["Gender"] == "Woman", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Non-binary", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Other (please describe)", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Man", "Gender"] = 'Man'
df.loc[df["Gender"] == "Decline to state", "Gender"] = 'Not_Man'

df.Gender.value_counts()

df.Q52.value_counts()
df['Religion'] = df.loc[:, 'Q52']
df.loc[df["Religion"] == "Agnostic (does not have a definite belief about whether God exists or not)", "Religion"] = 'No_Religion'
df.loc[df["Religion"] == "Nothing in particular", "Religion"] = 'No_Religion'
df.loc[df["Religion"] == "Muslim", "Religion"] = 'Other'
df.loc[df["Religion"] == "Decline to state", "Religion"] = 'Other'
df.loc[df["Religion"] == "Other (please describe)", "Religion"] = 'Other'
df.loc[df["Religion"] == "Christian - Catholic", "Religion"] = 'Christian'
df.loc[df["Religion"] == "Atheist (believes that God does not exist)", "Religion"] = 'No_Religion'
df.loc[df["Religion"] == "Christian - Protestant (e.g. Baptist, Lutheran, Methodist, Nondenominational, Presbyterian)", "Religion"] = 'Christian'
df.loc[df["Religion"] == "Christian - Church of Jesus Christ of Latter-day Saints", "Religion"] = 'Christian'
df.loc[df["Religion"] == "Jewish", "Religion"] = 'Other'
df.loc[df["Religion"] == "Hindu", "Religion"] = 'Other'
df.loc[df["Religion"] == "Buddhist", "Religion"] = 'Other'


df.Religion.value_counts()




df.QID40.value_counts()

# Asian / Pacific Islander
df['Race'] = df.loc[:, 'QID40']

df.loc[df["Race"] == "Other (please describe)", "Race"] = 'Other'
df.loc[df["Race"] == "Decline to state", "Race"] = 'Other'
df.loc[df["Race"] == "American Indian or Alaska Native", "Race"] = 'PEER'
df.loc[df["Race"] == "White/Caucasian", "Race"] = 'White'
df.loc[df["Race"] == "Hispanic, Latinx, or Spanish origin", "Race"] = 'PEER'
df.loc[df["Race"] == "Asian", "Race"] = 'Asian'
df.loc[df["Race"] == "Black or African American", "Race"] = 'PEER'
df.loc[df["Race"] == "Pacific Islander", "Race"] = 'PEER'

df.Race.value_counts()



df['Anxiety'] = df.loc[:, 'Q68']

df.Q68.value_counts()

df.loc[df["Anxiety"] == "Currently or having previously struggled with anxiety or an anxiety disorder", "Q68"] = 1
df.loc[df["Anxiety"] == "I have never struggled with an anxiety disorder", "Anxiety"] = 0
df.loc[df["Anxiety"] == "Decline to state", "Anxiety"] = 0
df.Anxiety.value_counts()



df['Depression'] = df.loc[:, 'Q70']

df.Q70.value_counts()
df.loc[df["Depression"] == "Currently or having previously struggled with depression or a depressive disorder", "Depression"] = 1
df.loc[df["Depression"] == "I have never struggled with depression", "Depression"] = 0
df.loc[df["Depression"] == "Decline to state", "Depression"] = 0
df.Depression.value_counts()

df = df[df.Course != 0]


df.Course.value_counts()
df.loc[df["Course"] == "B_2021", "Course"] = 'Unrevised'
df.loc[df["Course"] == "A_2022", "Course"] = 'Revised'
df.loc[df["Course"] == "B_2022", "Course"] = 'Revised'
df.Course.value_counts()

pre = df.loc[df['Pre_Post'] == 'Pre']
pre = pre.add_prefix('pre_')

post = df.loc[df['Pre_Post'] == 'Post']
post = post.add_prefix('post_')

df = pd.merge(pre, post, left_on='pre_RandomID', right_on='post_RandomID', how='left', indicator=True).drop('post_RandomID', axis=1)
df = df.loc[df['_merge'] == 'both']


# before filtering, send to Kelsey
tokelsey = pd.read_csv('tokelsey.csv')
tokelsey[-tokelsey['pre_RandomID'].isin(df['pre_RandomID'])]
tokelsey_new = df[-df['pre_RandomID'].isin(tokelsey['pre_RandomID'])]

tokelsey_new.to_csv("tokelsey_new.csv")

#
# I need to double check that this isn't doing something wrong with the null values

# one duplicate lacks all values in the pre survey for their "last" entry for the paired surveys. I'm dropping that one before running the drop_duplicates to keep their more informative response. 53714

df[df['pre_RandomID'] == 53714]
df = df.drop([317])
df[df['pre_RandomID'] == 53714]


dropthese = [15665, 38117, 38655, 60356]
df = df[-df.pre_RandomID.isin(dropthese)]

dups = df[df.duplicated(subset='pre_RandomID', keep=False)]

df = df.drop_duplicates(subset='pre_RandomID', keep="last")

dups_postremoval = df[df.duplicated(subset='pre_RandomID', keep=False)]

# Drop a guy who took the survey a bunch and his answers are all over the place df[df['pre_RandomID'] == 15665]
# Also drop these, who have entries for multiple courses: 38117, 38655, 60356



# 10 individuals had different pre-post answers. All were recoded as "LGBTQ"

conditions = [df['pre_LGBTQ'] == df['post_LGBTQ'], df['pre_LGBTQ'] != df['post_LGBTQ']]
choices = [df['pre_LGBTQ'], 'LGBTQ']
df['LGBTQ'] = np.select(conditions, choices)

# 1 individual didn't answer this question in post. They identified as "Man" in the pre course survey and were recoded as Man

conditions = [df['pre_Gender'] == df['post_Gender'], df['pre_Gender'] != df['post_Gender']]
choices = [df['pre_Gender'], 'Man']
df['Gender'] = np.select(conditions, choices)

# 44 individuals have different pre-post responses for religion
# Q52

x = (df[df['pre_Religion'] != df['post_Religion']]).groupby(['pre_Q52', 'post_Q52']).size()
y = (df[df['pre_Religion'] != df['post_Religion']]).groupby(['pre_Religion', 'post_Religion']).size()
# x.to_csv('religion.csv')
# y.to_csv('religion_2.csv')

# Here, edit the file by hand to add in a "Recode" column
df[df['pre_Religion'] != df['post_Religion']]
df[df['pre_Religion'] != df['post_Religion']]['pre_Religion']
df[df['pre_Religion'] != df['post_Religion']]['post_Religion']

r = pd.read_csv('religion.csv')

for index, row in r.iterrows():
    df.loc[(df['pre_Religion'] != df['post_Religion']) & (df['pre_Q52'] == row[f"pre_Q52"]) & (df['post_Q52'] == row[f"post_Q52"]), 'pre_Religion'] = row[f"Recode"]
    df.loc[(df['pre_Religion'] != df['post_Religion']) & (df['pre_Q52'] == row[f"pre_Q52"]) & (df['post_Q52'] == row[f"post_Q52"]), 'post_Religion'] = row[f"Recode"]

# Changing the individual who listed both "No Religion" and "Christian" to "Christian"
df.loc[(df['pre_Religion'] != df['post_Religion']), 'post_Religion'] = "Christian"



conditions = [df['pre_Religion'] == df['post_Religion'], (df['pre_Religion'] != df['post_Religion']) & (df['pre_Religion'] == 'Other'), (df['pre_Religion'] != df['post_Religion']) & (df['post_Religion'] == 'Other'), (df['pre_Religion'] != df['post_Religion']) & (df['post_Religion'] != 'Other') & (df['pre_Religion'] != 'Other')]
choices = [df['pre_Religion'], df['post_Religion'], df['pre_Religion'], 'Other']
df['Religion'] = np.select(conditions, choices)

df['Religion'].value_counts()





x = (df[df['pre_Race'] != df['post_Race']]).groupby(['pre_QID40', 'post_QID40']).size()
y = (df[df['pre_Race'] != df['post_Race']]).groupby(['pre_Race', 'post_Race']).size()


x.to_csv('race.csv')
y.to_csv('race_2.csv')

# Here, edit the file by hand to add in a "Recode" column
df[df['pre_Race'] != df['post_Race']]
df[df['pre_Race'] != df['post_Race']]['pre_Religion']
df[df['pre_Race'] != df['post_Race']]['post_Religion']

r = pd.read_csv('race.csv')

conditions = [(df['pre_Race'] == df['post_Race']), (df['pre_Race'] != df['post_Race']) & (df['pre_Race'] == 'Other'), (df['pre_Race'] != df['post_Race']) & (df['post_Race'] == 'Other')]
choices = [df['pre_Race'], df['post_Race'], df['pre_Race']]
df['Race'] = np.select(conditions, choices)

df.loc[df["Race"] == 0, "pre_QID40"]
df.loc[df["Race"] == 0, "post_QID40"]

conditions = [df['pre_QID40_PEER'] == df['post_QID40_PEER'], df['pre_QID40_PEER'] != df['post_QID40_PEER']]
choices = [df['pre_QID40_PEER'], 1]
df['Race_PEER'] = np.select(conditions, choices)

conditions = [df['pre_Q68'] == df['post_Q68'], df['pre_Q68'] != df['post_Q68']]
choices = [df['pre_Q68'], 1]
df['Anxiety'] = np.select(conditions, choices)

conditions = [df['pre_Q70'] == df['post_Q70'], df['pre_Q70'] != df['post_Q70']]
choices = [df['pre_Q70'], 1]
df['Depression'] = np.select(conditions, choices)


conditions = [df['Anxiety'] == df['Depression'], df['Anxiety'] != df['Depression']]
choices = [df['Anxiety'], 1]
df['AnxietyDepression'] = np.select(conditions, choices)


df = df.replace('Strongly disagree', float(-3))
df = df.replace('Disagree', float(-2))
df = df.replace('Slightly disagree', float(-1))
df = df.replace('Strongly agree', float(3))
df = df.replace('Agree', float(2))
df = df.replace('Slightly agree', float(1))
df = df.replace('I am unsure', float(0))

df = df.replace('5%', float(0.05))
df = df.replace('20%', float(0.2))
df = df.replace('33%', float(0.33))
df = df.replace('66%', float(0.66))
df = df.replace('80%', float(0.8))
df = df.replace('95%', float(0.95))
df = df.replace('100%', float(1))
df = df.replace('I am unsure', float(0))
df = df.replace('0%, I do not think that homosexual behaviors occur outside of humans', float(0))
df = df.replace('0%, I do not think this occurs outside of humans', float(0))




# Create a column that is post_Q16 - pre_Q16
list = ['Q16', 'Q21', 'Q1', 'Q3', 'Q58', 'Q9', 'Q7', 'Q8', 'Q10', 'Q11', 'Q13', 'Q14', 'Q17', 'Q18', 'Q19', 'Q2', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']


for i in list:
    df['diff_' + i] = df['post_' + i] - df['pre_' + i]




# pd.CategoricalDtype(categories=['Strongly disagree', 'Disagree', 'Slightly disagree', 'I am unsure', 'Slightly agree', 'Agree', 'Strongly agree'], ordered=True)

pd.CategoricalDtype(categories=['-3', '-2', '-1', '0', '1', '2', '3'], ordered=True)

LGBTQ, Gender, Religion_Christian, Religion_Atheist, Religion_Muslim, Race_Asian, Race_PEER, AnxietyDepression


df['LGBTQ'].value_counts().to_csv('LGBTQ_Counts.csv')
df['Gender'].value_counts().to_csv('Gender_Counts.csv')
df['Religion_Christian'].value_counts().to_csv('Religion_Christian_Counts.csv')
df['Religion_Atheist'].value_counts().to_csv('Religion_Atheist_Counts.csv')
df['Religion_Muslim'].value_counts().to_csv('Religion_Muslim_Counts.csv')
df['Race_Asian'].value_counts().to_csv('Race_Asian_Counts.csv')
df['Race_PEER'].value_counts().to_csv('Race_PEER_Counts.csv')
df['AnxietyDepression'].value_counts().to_csv('AnxietyDepression_Counts.csv')
df['pre_Course'].value_counts().to_csv('Course_Counts.csv')

cattype = pd.CategoricalDtype(categories=[-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2., 3.0, 4.0, 5.0, 6.0], ordered=True)

df['Course'] = df.loc[:, 'pre_Course']


list = ['Q16', 'Q21', 'Q1', 'Q3', 'Q58', 'Q9', 'Q7', 'Q8', 'Q10', 'Q11', 'Q13', 'Q14', 'Q17', 'Q18', 'Q19', 'Q2', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']

for i in list:
    df2 = df[df['diff_' + i].notna()]
    df2['Course'] = df2.loc[:, 'pre_Course']
    df2['diff_' + i].astype(cattype)
    mod_log = OrderedModel(df2['diff_' + i], df2[['Course', 'LGBTQ', 'Gender', 'Religion_Christian', 'Religion_Atheist', 'Religion_Muslim', 'Race_Asian', 'Race_PEER', 'AnxietyDepression']], distr='logit')
    res_log = mod_log.fit(method='bfgs', disp=False)
    res_log.summary()
    with open('log/' + i + '_log_summary.csv', 'w') as fh:
        fh.write(res_log.summary().as_csv())


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


# Modeling with interactions
# in python

import statsmodels.formula.api as smf

df2['Course'] = df2.loc[:, 'pre_Course']
df['Race'] = df.loc[:, 'post_QID40']
df['Religion'] = df.loc[:, 'pre_Q52']
df['LGBTQ_linear'] = df.loc[:, 'pre_Q54']
df['Gender_linear'] = df.loc[:, 'pre_Q40']


list = ['Q16', 'Q21', 'Q1', 'Q3', 'Q58', 'Q9', 'Q7', 'Q8', 'Q10', 'Q11', 'Q13', 'Q14', 'Q17', 'Q18', 'Q19', 'Q2', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']

sig_2 = pd.DataFrame(columns=[list], index = df4.index)
direction_2 = pd.DataFrame(columns=[list], index = df4.index)

# model = smf.ols(formula=f'diff_{i} ~ Course + LGBTQ + Gender + Race + AnxietyDepression + Religion + Course*LGBTQ + Course*Gender + Course*Religion  + Course*AnxietyDepression', data=df).fit()

# collapse race into one column and include course*race

for i in list:
model = smf.ols(formula=f'diff_{i} ~ Course + LGBTQ_linear + c"Gender" + Race_Asian + Race_PEER + AnxietyDepression + Religion + Course*LGBTQ_linear + Course*Gender + Course*Religion + Course*AnxietyDepression', data=df).fit()
model.summary()
with open('linear/' + i + '_summary.csv', 'w') as fh:
    fh.write(model.summary().as_csv())
model_summary = model.summary()
model_as_html = model_summary.tables[1].as_html()
df4 = pd.read_html(model_as_html, header=0, index_col=0)[0]
s = df4['P>|t|'] < 0.05
sig_2[i] = s
q = df4['coef'] < 0
q.index = direction_2.index
direction_2[i] = q

direction_2 = direction_2.astype(int)
direction_2 = direction_2 * -1
direction_2 = direction_2.replace(0,1)
sig_2 = sig_2.astype(int)
sig_dir = direction_2 * sig_2

sig_dir_t = sig_dir.transpose()
sig_dir_t.to_csv('linear/linear_sig_direction.csv')
df.to_csv('tokelsey.csv')

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
