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
df.loc[df["LGBTQ"] == "No", "LGBTQ"] = '0_Not_LGBTQ'
df.loc[df["LGBTQ"] == "Decline to state", "LGBTQ"] = '0_Not_LGBTQ'

# Self identified as LGBTQ, Other, "Straight, ally"
df.loc[df["RandomID"] == 41529, "LGBTQ"] = '0_Not_LGBTQ'

df.LGBTQ.value_counts()

df.Q40.value_counts()
df['Gender'] = df.loc[:, 'Q40']

df.loc[df["Gender"] == "Woman", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Non-binary", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Other (please describe)", "Gender"] = 'Not_Man'
df.loc[df["Gender"] == "Man", "Gender"] = '0_Man'
df.loc[df["Gender"] == "Decline to state", "Gender"] = 'Not_Man'

df.Gender.value_counts()

df.Q52.value_counts()

df['Religion'] = df.loc[:, 'Q52']
df.loc[df["Religion"] == "Agnostic (does not have a definite belief about whether God exists or not)", "Religion"] = '0_No_Religion'

df.loc[df["Religion"] == "Nothing in particular", "Religion"] = '0_No_Religion'
df.loc[df["Religion"] == "Muslim", "Religion"] = 'Abrahamic'
df.loc[df["Religion"] == "Decline to state", "Religion"] = 'Other'
df.loc[df["Religion"] == "Other (please describe)", "Religion"] = 'Other'
df.loc[df["Religion"] == "Christian - Catholic", "Religion"] = 'Abrahamic'
df.loc[df["Religion"] == "Atheist (believes that God does not exist)", "Religion"] = '0_No_Religion'
df.loc[df["Religion"] == "Christian - Protestant (e.g. Baptist, Lutheran, Methodist, Nondenominational, Presbyterian)", "Religion"] = 'Abrahamic'
df.loc[df["Religion"] == "Christian - Church of Jesus Christ of Latter-day Saints", "Religion"] = 'Abrahamic'
df.loc[df["Religion"] == "Jewish", "Religion"] = 'Abrahamic'
df.loc[df["Religion"] == "Hindu", "Religion"] = 'Other'
df.loc[df["Religion"] == "Buddhist", "Religion"] = 'Other'


df.Religion.value_counts()




df.QID40.value_counts()

# Asian / Pacific Islander
df['Race'] = df.loc[:, 'QID40']

df.loc[df["Race"] == "Other (please describe)", "RandomID"]
df.loc[df["Race"] == "Other (please describe)", "QID40_7_TEXT"]

# 24894 = PEER
# 7921 = Asian
# 90442 = Asian
# 98137 = PEER
# 35840 = Asian
# 51650 = PEER
# 42298 = Asian
# 84156 = PEER
# 85958 = NA
# 8216 = Asian
# 80621 = Asian
# 5501 = PEER

df.loc[df["RandomID"] == 24894, "Race"] = 'PEER'
df.loc[df["RandomID"] == 7921, "Race"] = 'Asian'
df.loc[df["RandomID"] == 90442, "Race"] = 'Asian'
df.loc[df["RandomID"] == 98137, "Race"] = 'PEER'
df.loc[df["RandomID"] == 35840, "Race"] = 'Asian'
df.loc[df["RandomID"] == 51650, "Race"] = 'PEER'
df.loc[df["RandomID"] == 42298, "Race"] = 'Asian'
df.loc[df["RandomID"] == 84156, "Race"] = 'PEER'
df.loc[df["RandomID"] == 85958, "Race"] = 'NA'
df.loc[df["RandomID"] == 8216, "Race"] = 'Asian'
df.loc[df["RandomID"] == 80621, "Race"] = 'Asian'
df.loc[df["RandomID"] == 5501, "Race"] = 'PEER'

df.loc[df["Race"] == "Other (please describe)", "Race"] = 'NA'
df.loc[df["Race"] == "Decline to state", "Race"] = 'NA'
df.loc[df["Race"] == "American Indian or Alaska Native", "Race"] = 'PEER'
df.loc[df["Race"] == "White/Caucasian", "Race"] = '0_White'
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
df.loc[df["Course"] == "B_2021", "Course"] = '0_Unrevised'
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
# tokelsey = pd.read_csv('tokelsey.csv')
# tokelsey[-tokelsey['pre_RandomID'].isin(df['pre_RandomID'])]
# tokelsey_new = df[-df['pre_RandomID'].isin(tokelsey['pre_RandomID'])]

# cleanData = pd.read_csv('cleanData.csv')
# cleanData[-cleanData['RandomID'].isin(df['pre_RandomID'])]
# df[-df['pre_RandomID'].isin(cleanData['RandomID'])]
# tokelsey_new = df[-df['pre_RandomID'].isin(tokelsey['pre_RandomID'])]
# tokelsey_new.to_csv("tokelsey_new.csv")

# one duplicate lacks all values in the pre survey for their "last" entry for the paired surveys. I'm dropping that one before running the drop_duplicates to keep their more informative response. 53714.
# Another duplicate lacks all values in their post survey for their "last" entry, so again I have to drop the last one to keep their more informative response. 89739

df[df['pre_RandomID'] == 53714]
df = df.drop([317])
df[df['pre_RandomID'] == 53714]

df[df['pre_RandomID'] == 89739]
df = df.drop([282])
df[df['pre_RandomID'] == 89739]



dropthese = [15665, 38117, 38655, 60356, 22207]
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
choices = [df['pre_Gender'], '0_Man']
df['Gender'] = np.select(conditions, choices)

# 44 individuals have different pre-post responses for religion
# Q52

x = (df[df['pre_Religion'] != df['post_Religion']]).groupby(['pre_Q52', 'post_Q52']).size()
y = (df[df['pre_Religion'] != df['post_Religion']]).groupby(['pre_Religion', 'post_Religion']).size()
# x.to_csv('religion.csv')
# y.to_csv('religion_2.csv')

# Here, edit the file by hand to add in a "Recode" column
df[df['pre_Religion'] != df['post_Religion']]
df.loc[df['pre_Religion'] != df['post_Religion']][['pre_RandomID', 'pre_Q52', 'post_Q52', 'pre_Q52_12_TEXT']]
df[df['pre_Religion'] != df['post_Religion']]['post_Religion']

#this person said they were raised Catholic but are no longer religions
df.loc[df["pre_RandomID"] == 11517, "Religion"] = 'Abrahamic'



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


# x.to_csv('race.csv')
# y.to_csv('race_2.csv')

# Here, edit the file by hand to add in a "Recode" column
df[df['pre_Race'] != df['post_Race']]
df[df['pre_Race'] != df['post_Race']]['pre_Race']
df[df['pre_Race'] != df['post_Race']]['post_Race']

# Edit race .csv to add in recode column
r = pd.read_csv('race.csv')

for index, row in r.iterrows():
    df.loc[(df['pre_Race'] != df['post_Race']) & (df['pre_QID40'] == row[f"pre_QID40"]) & (df['post_QID40'] == row[f"post_QID40"]), 'pre_Race'] = row[f"Recode"]
    df.loc[(df['pre_Race'] != df['post_Race']) & (df['pre_QID40'] == row[f"pre_QID40"]) & (df['post_QID40'] == row[f"post_QID40"]), 'post_Race'] = row[f"Recode"]

# One individual remains with mismatched pre and post race. Their pre is PEER and their post is NaN so I'm changing the post to match
df.loc[df['pre_Race'] != df['post_Race'], 'post_Race'] = 'PEER'






conditions = [(df['pre_Race'] == df['post_Race']), (df['pre_Race'] != df['post_Race']) & (df['pre_Race'] == 'Other'), (df['pre_Race'] != df['post_Race']) & (df['post_Race'] == 'Other')]
choices = [df['pre_Race'], df['post_Race'], df['pre_Race']]
df['Race'] = np.select(conditions, choices)

# Lump 'Asian' with 'Other' due to small sample sizes
# df.loc[df["Race"] == "Asian", "Race"] = 'Other'


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
df = df.replace('I am unsure', 'NA')
df = df.replace('0%, I do not think that homosexual behaviors occur outside of humans', float(0))
df = df.replace('0%, I do not think this occurs outside of humans', float(0))


df['Course'] = df.loc[:, 'pre_Course']


# Create a column that is post_Q16 - pre_Q16
list = ['Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q58', 'Q37_1', 'Q37_2', 'Q37_3', 'Q37_4', 'Q37_5', 'Q38_1', 'Q38_2', 'Q38_3', 'Q38_4', 'Q38_5']


for i in list:
    df['diff_' + i] = df['post_' + i] - df['pre_' + i]

df.to_csv('cleanData_danny.csv')
