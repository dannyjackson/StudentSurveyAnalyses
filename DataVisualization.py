# data visualization
# cd Documents/AnBeh_Revised/SurveyResults

# In command line:
#conda create -c conda-forge -n spyder-env spyder numpy scipy pandas matplotlib sympy cython

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data_Nov22.csv')

plt.style.use('seaborn-deep')

x = df.loc[df.Course == '0_Unrevised', 'diff_Q1']
y = df.loc[df.Course == 'Revised', 'diff_Q1']




list = ['Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q37_total', 'Q38_total']

yticks = np.arange(0, 200, 10)

# on stacked plots
for i in list:
    x = df.loc[df.Course == '0_Unrevised', f'diff_{i}']
    y = df.loc[df.Course == 'Revised', f'diff_{i}']
    fig, ((hist_unrevised, hist_revised)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    hist_unrevised.hist(x, color = 'salmon', bins = [-6.5, -5.5, -4.5, -3.5, -2.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], label='Unrevised')
    hist_unrevised.axvline(x.mean(), color = 'brown', linestyle='dashed', linewidth=1)
    hist_unrevised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_unrevised.legend(loc='upper right')
    hist_unrevised.set_xlim([-6, 6])
    hist_unrevised.set_yticks(yticks)
    hist_unrevised.set_ylim([0, x.value_counts().max() + 5])
    hist_unrevised.axes.get_xaxis().set_visible(False)
    hist_unrevised.set_title(f'{i}: Change in Survey Response (Post - Pre)')
    hist_revised.hist(y, color = 'darkcyan', bins = [-6.5, -5.5, -4.5, -3.5, -2.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], label='Revised')
    hist_revised.axvline(y.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
    hist_revised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_revised.legend(loc='upper right')
    hist_revised.set_xlim([-6, 6])
    hist_revised.set_yticks(yticks)
    hist_revised.set_ylim([0, y.value_counts().max() + 5])
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(f'figures/diff_{i}.png', bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

list2 = ['Q13', 'Q58']

for i in list2:
    x = df.loc[df.Course == '0_Unrevised', f'diff_{i}']
    y = df.loc[df.Course == 'Revised', f'diff_{i}']
    fig, ((hist_unrevised, hist_revised)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    hist_unrevised.hist(x, color = 'salmon', bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], label='Unrevised')
    hist_unrevised.axvline(x.mean(), color = 'brown', linestyle='dashed', linewidth=1)
    hist_unrevised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_unrevised.legend(loc='upper right')
    hist_unrevised.set_xlim([-1, 1])
    hist_unrevised.set_yticks(yticks)
    hist_unrevised.set_ylim([0, x.value_counts().max() + 5])
    hist_unrevised.axes.get_xaxis().set_visible(False)
    hist_unrevised.set_title(f'{i}: Change in Survey Response (Post - Pre)')
    hist_revised.hist(y, color = 'darkcyan', bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], label='Revised')
    hist_revised.axvline(y.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
    hist_revised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_revised.legend(loc='upper right')
    hist_revised.set_xlim([-1, 1])
    hist_revised.set_yticks(yticks)
    hist_revised.set_ylim([0, y.value_counts().max() + 5])
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(f'figures/diff_{i}.png', bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

list3 = ['Q37_total']

for i in list3:
    x = df.loc[df.Course == '0_Unrevised', f'diff_{i}']
    y = df.loc[df.Course == 'Revised', f'diff_{i}']
    fig, ((hist_unrevised, hist_revised)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    hist_unrevised.hist(x, color = 'salmon', label='Unrevised')
    hist_unrevised.axvline(x.mean(), color = 'brown', linestyle='dashed', linewidth=1)
    hist_unrevised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_unrevised.legend(loc='upper right')
    hist_unrevised.set_xlim([-28, 5])
    hist_unrevised.set_yticks(yticks)
    hist_unrevised.set_ylim([0, x.value_counts().max() + 5])
    hist_unrevised.axes.get_xaxis().set_visible(False)
    hist_unrevised.set_title(f'{i}: Change in Survey Response (Post - Pre)')
    hist_revised.hist(y, color = 'darkcyan', label='Revised')
    hist_revised.axvline(y.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
    hist_revised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_revised.legend(loc='upper right')
    hist_revised.set_xlim([-28, 5])
    hist_revised.set_yticks(yticks)
    hist_revised.set_ylim([0, y.value_counts().max() + 5])
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(f'figures/diff_{i}.png', bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

list4 = ['Q38_total']
for i in list4:
    x = df.loc[df.Course == '0_Unrevised', f'diff_{i}']
    y = df.loc[df.Course == 'Revised', f'diff_{i}']
    fig, ((hist_unrevised, hist_revised)) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    hist_unrevised.hist(x, color = 'salmon', label='Unrevised')
    hist_unrevised.axvline(x.mean(), color = 'brown', linestyle='dashed', linewidth=1)
    hist_unrevised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_unrevised.legend(loc='upper right')
    hist_unrevised.set_xlim([-21, 16])
    hist_unrevised.set_yticks(yticks)
    hist_unrevised.set_ylim([0, x.value_counts().max() + 5])
    hist_unrevised.axes.get_xaxis().set_visible(False)
    hist_unrevised.set_title(f'{i}: Change in Survey Response (Post - Pre)')
    hist_revised.hist(y, color = 'darkcyan', label='Revised')
    hist_revised.axvline(y.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
    hist_revised.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
    hist_revised.legend(loc='upper right')
    hist_revised.set_xlim([-21, 16])
    hist_revised.set_yticks(yticks)
    hist_revised.set_ylim([0, y.value_counts().max() + 5])
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(f'figures/diff_{i}.png', bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

df.value_counts(["Course", "LGBTQ"])
df.value_counts(["Course", "pre_Q54"])
df.value_counts(["Course", "post_Q54"])

df.value_counts(["Course", "Gender"])
df.value_counts(["Course", "pre_Q40"])
df.value_counts(["Course", "post_Q40"])

df.value_counts(["Course", "Religion"])
df.value_counts(["Course", "pre_Q52"])
df.value_counts(["Course", "post_Q52"])

df.value_counts(["Course", "Race"])
df.value_counts(["Course", "pre_QID40"])
df.value_counts(["Course", "post_QID40"])

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

x_all = df['pre_Q38_total']
x_lgbtq = df.loc[df["LGBTQ"] == "LGBTQ", "pre_Q38_total"]
x_gender = df.loc[df["Gender"] == "Not_Man", "pre_Q38_total"]
x_religion = df.loc[df["Religion"] == "Abrahamic", "pre_Q38_total"]
x_race = df.loc[df["Race"] == "PEER", "pre_Q38_total"]


fig, ((hist_all, hist_lgbtq, hist_gender, hist_religion, hist_race)) = plt.subplots(nrows=5, ncols=1, figsize=(5, 8))
plt.subplots_adjust(wspace=0, hspace=0)

hist_all.hist(x_all, color = 'lightgray', edgecolor='black', label = "All students")
hist_all.axvline(x_all.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_all.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_all.legend(loc='upper left')
hist_all.set_xlim([-6, 15])
hist_all.axes.get_xaxis().set_visible(False)
hist_all.set_title('Pre-Course Sense of Inclusion in Biology')

hist_lgbtq.hist(x_lgbtq, color = 'lightgray', edgecolor='black', label = "LGBTQ students")
hist_lgbtq.axvline(x_lgbtq.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_lgbtq.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_lgbtq.legend(loc='upper left')
hist_lgbtq.set_xlim([-6, 15])
hist_lgbtq.axes.get_xaxis().set_visible(False)

hist_gender.hist(x_gender, color = 'lightgray', edgecolor='black', label = "Not man students")
hist_gender.axvline(x_gender.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_gender.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_gender.legend(loc='upper left')
hist_gender.set_xlim([-6, 15])
hist_gender.axes.get_xaxis().set_visible(False)

hist_religion.hist(x_religion, color = 'lightgray', edgecolor='black', label = "Christian, Jewish, and Muslim students")
hist_religion.axvline(x_religion.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_religion.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_religion.legend(loc='upper left')
hist_religion.set_xlim([-6, 15])
hist_religion.axes.get_xaxis().set_visible(False)

hist_race.hist(x_race, color = 'lightgray', edgecolor='black', label = "PEER students")
hist_race.axvline(x_race.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_race.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_race.legend(loc='upper left')
hist_race.set_xlim([-6, 15])

plt.savefig(f'figures/pre38.png', bbox_inches='tight')




plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

x_all = df['pre_Q37_total']
x_lgbtq = df.loc[df["LGBTQ"] == "LGBTQ", "pre_Q37_total"]
x_gender = df.loc[df["Gender"] == "Not_Man", "pre_Q37_total"]
x_religion = df.loc[df["Religion"] == "Abrahamic", "pre_Q37_total"]
x_race = df.loc[df["Race"] == "PEER", "pre_Q37_total"]


fig, ((hist_all, hist_lgbtq, hist_gender, hist_religion, hist_race)) = plt.subplots(nrows=5, ncols=1, figsize=(5, 8))
plt.subplots_adjust(wspace=0, hspace=0)

hist_all.hist(x_all, color = 'lightgray', edgecolor='black', label = "All students")
hist_all.axvline(x_all.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_all.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_all.legend(loc='upper left')
hist_all.set_xlim([0, 15])
hist_all.axes.get_xaxis().set_visible(False)
hist_all.set_title('Pre-Course Sense of Inclusion in This Course')

hist_lgbtq.hist(x_lgbtq, color = 'lightgray', edgecolor='black', label = "LGBTQ students")
hist_lgbtq.axvline(x_lgbtq.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_lgbtq.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_lgbtq.legend(loc='upper left')
hist_lgbtq.set_xlim([0, 15])
hist_lgbtq.axes.get_xaxis().set_visible(False)

hist_gender.hist(x_gender, color = 'lightgray', edgecolor='black', label = "Not man students")
hist_gender.axvline(x_gender.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_gender.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_gender.legend(loc='upper left')
hist_gender.set_xlim([0, 15])
hist_gender.axes.get_xaxis().set_visible(False)

hist_religion.hist(x_religion, color = 'lightgray', edgecolor='black', label = "Christian, Jewish, and Muslim students")
hist_religion.axvline(x_religion.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_religion.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_religion.legend(loc='upper left')
hist_religion.set_xlim([0, 15])
hist_religion.axes.get_xaxis().set_visible(False)

hist_race.hist(x_race, color = 'lightgray', edgecolor='black', label = "PEER students")
hist_race.axvline(x_race.mean(), color = 'darkslategray', linestyle='dashed', linewidth=1)
hist_race.axvline(0, color = 'Black', linestyle='solid', linewidth=0.5)
hist_race.legend(loc='upper left')
hist_race.set_xlim([0, 15])

plt.savefig(f'figures/pre37.png', bbox_inches='tight')


model = smf.ols(formula=f'pre_Q37_total ~  LGBTQ + Gender + Race + Religion', data=df).fit()

sm.stats.anova_lm(model, typ=2)
>>> sm.stats.anova_lm(model, typ=2)
               sum_sq     df         F    PR(>F)
LGBTQ       41.365957    1.0  4.240505  0.040799
Gender       0.164405    1.0  0.016853  0.896842
Race        21.042404    1.0  2.157098  0.143524
Religion    16.462954    1.0  1.687649  0.195445
Residual  1902.217377  195.0       NaN       NaN

model = smf.ols(formula=f'pre_Q38_total ~  LGBTQ + Gender + Race + Religion', data=df).fit()

sm.stats.anova_lm(model, typ=2)

               sum_sq     df         F    PR(>F)
LGBTQ       81.093764    1.0  5.350255  0.021762
Gender       1.794897    1.0  0.118420  0.731125
Race        14.759172    1.0  0.973753  0.324969
Religion     1.204682    1.0  0.079480  0.778302
Residual  2955.613075  195.0       NaN       NaN







# looking into qualitative responses from LGBTQ students who changed their sense of inclusion
improvedLGBTQ = df.loc[df["diff_Q38_total"] > 0]
improvedLGBTQ = improvedLGBTQ.loc[improvedLGBTQ["LGBTQ"] == "LGBTQ"]
improvedLGBTQ_revised = improvedLGBTQ.loc[improvedLGBTQ["Course"] == "Revised"]
improvedLGBTQ_unrevised = improvedLGBTQ.loc[improvedLGBTQ["Course"] == "0_Unrevised"]
improvedLGBTQ_revised.to_csv("improvedLGBTQ_revised.csv")
improvedLGBTQ_unrevised.to_csv("improvedLGBTQ_unrevised.csv")




import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data_Nov22.csv')

df_gay = df.loc[df["LGBTQ"] == "LGBTQ"]
df_melted_3 = pd.melt(df_gay, id_vars=['pre_RandomID', 'Course',], value_vars=["diff_Q38_total"])

df_melted_3.columns = ['pre_RandomID', 'Course', 'Question', 'Difference in Response Score']
df_melted_3.loc[df_melted_3.Question == 'diff_Q38_total', 'Question'] = 'Q38'


plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

sns.set(rc={'figure.figsize':(4,1)})
sns.set_theme(style='white')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
bx = sns.boxplot(x="Difference in Response Score", y="Question", hue="Course", fliersize = 0.1, data=df_melted_3.reset_index())
sns.move_legend(bx, "upper left", bbox_to_anchor=(1, 1))
plt.show()

plt.savefig(f'figures/boxplot_inclusion.png', bbox_inches='tight')



df_melted_2 = pd.melt(df, id_vars=['pre_RandomID', 'Course',], value_vars=["diff_Q58", "diff_Q13"])

df_melted_2.columns = ['pre_RandomID', 'Course', 'Question', 'Difference in Response Score']
df_melted_2.loc[df_melted_2.Question == 'diff_Q13', 'Question'] = 'Q13'
df_melted_2.loc[df_melted_2.Question == 'diff_Q58', 'Question'] = 'Q58'


plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

sns.set(rc={'figure.figsize':(4,1)})
sns.set_theme(style='white')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
bx = sns.boxplot(x="Difference in Response Score", y="Question", hue="Course", fliersize = 0.1, data=df_melted_2.reset_index())
sns.move_legend(bx, "upper left", bbox_to_anchor=(1, 1))
plt.show()

plt.savefig(f'figures/boxplot_prop.png', bbox_inches='tight')




df_melted = pd.melt(df, id_vars=['pre_RandomID', 'Course',], value_vars=["diff_Q1", "diff_Q2", "diff_Q3", "diff_Q11",  "diff_Q9",  "diff_Q14", "diff_Q16", "diff_Q19"])

df_melted.columns = ['pre_RandomID', 'Course', 'Question', 'Difference in Response Score']

df_melted.loc[df_melted.Question == 'diff_Q1', 'Question'] = 'Q1'
df_melted.loc[df_melted.Question == 'diff_Q2', 'Question'] = 'Q2'
df_melted.loc[df_melted.Question == 'diff_Q3', 'Question'] = 'Q3'
df_melted.loc[df_melted.Question == 'diff_Q11', 'Question'] = 'Q11'
df_melted.loc[df_melted.Question == 'diff_Q9', 'Question'] = 'Q9'

df_melted.loc[df_melted.Question == 'diff_Q14', 'Question'] = 'Q14'
df_melted.loc[df_melted.Question == 'diff_Q16', 'Question'] = 'Q16'
df_melted.loc[df_melted.Question == 'diff_Q19', 'Question'] = 'Q19'


plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

sns.set(rc={'figure.figsize':(4,8)})
sns.set_theme(style='white')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
bx = sns.boxplot(x="Difference in Response Score", y="Question", hue="Course", fliersize = 0.1, data=df_melted.reset_index())
sns.move_legend(bx, "upper left", bbox_to_anchor=(1, 1))


plt.savefig(f'figures/boxplot.png', bbox_inches='tight')







df_gay = df.loc[df["LGBTQ"] == "LGBTQ"]
df_notgay = df.loc[df["LGBTQ"] == "0_Not_LGBTQ"]


list = ['Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q58', 'Q37_total', 'Q38_total']


for i in list:
    print(i)
    df[f'pre_{i}'].mean()
    df_gay[f'pre_{i}'].mean()
    df_notgay[f'pre_{i}'].mean()
    ttest_ind(df_gay[f'pre_{i}'], df_notgay[f'pre_{i}'], nan_policy='omit')



rows = ['Intercept', 'LGBTQ[T.LGBTQ]', 'Gender[T.Not_Man]', 'Religion[T.Abrahamic]']
sig = pd.DataFrame(columns=[list], index = rows)
coefficient = pd.DataFrame(columns=[list], index = rows)
sig_all = pd.DataFrame(columns=[list], index = rows)

rows_anova = ['LGBTQ', 'Gender', 'Religion',
       'Residual']
sig_anova = pd.DataFrame(columns=[list], index = rows_anova)
coefficient_anova = pd.DataFrame(columns=[list], index = rows_anova)

for i in list:
    df[f'diff_{i}'].mean()
    model = smf.ols(formula=f'pre_{i} ~ LGBTQ + Gender + Religion', data=df).fit()
    model.summary()
    with open('linear/pre_' + i + '_summary.csv', 'w') as fh:
        fh.write(model.summary().as_csv())
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv('linear/pre_' + i + '_anova_summary.csv')
    model_summary = model.summary()
    model_as_html = model_summary.tables[1].as_html()
    df2 = pd.read_html(model_as_html, header=0, index_col=0)[0]
    s = df2['P>|t|'] < 0.05
    sig[i] = s
    sig.loc['Intercept_all', i] = 1
    s_all = df2['P>|t|']
    sig_all[i] = s_all
    t = df2['coef']['Intercept']
    q = df2['coef']
    sa = anova_table['PR(>F)']
    sig_anova[i] = sa
    ca = anova_table['sum_sq']
    coefficient_anova[i] = ca
    coefficient[i] = q
    coefficient.loc['Intercept_all', i] = t



sig = sig.astype(int)
sig_dir = coefficient * sig

sig_anova_t = sig_anova.transpose()
sig_anova_t.to_csv('linear/pre_anova_sig.csv')
coefficient_anova_t = coefficient_anova.transpose()
coefficient_anova_t.to_csv('linear/pre_anova_coef.csv')

sig_dir_t = sig_dir.transpose()
sig_dir_t.to_csv('linear/pre_linear_sig_direction.csv')

sig_all_t = sig_all.transpose()
sig_all_t.to_csv('linear/pre_sig_all.csv')
coefficient_t = coefficient.transpose()
coefficient_t.to_csv('linear/pre_coefficient.csv')

coefficient_t = coefficient_t.drop(['Intercept', 'Intercept_all'], axis=1)
sig_anova_t = sig_anova_t.drop('Residual', axis=1)
coefficient_t.columns = sig_anova_t.columns
sig_coeff_anovalm = (sig_anova_t < 0.05).astype(int) * coefficient_t

sig_coeff_anovalm.to_csv("linear/pre_sig_coeff_anovalm.csv")




# PCA


import pandas as pd

df = pd.read_csv('data_Nov22.csv')

from sklearn.preprocessing import StandardScaler
features_all = ['post_Q1', 'post_Q2', 'post_Q3', 'post_Q7', 'post_Q8', 'post_Q9', 'post_Q10', 'post_Q11', 'post_Q14', 'post_Q16', 'post_Q17', 'post_Q18', 'post_Q19', 'post_Q21', 'post_Q22', 'post_Q24', 'post_Q25', 'post_Q27', 'post_Q28', 'post_Q30', 'post_Q31', 'post_Q33', 'post_Q34', 'LGBTQ', 'Religion', 'Gender']

features = ['post_Q1', 'post_Q2', 'post_Q3', 'post_Q7', 'post_Q8', 'post_Q9', 'post_Q10', 'post_Q11', 'post_Q14', 'post_Q16', 'post_Q17', 'post_Q18', 'post_Q19', 'post_Q21', 'post_Q22', 'post_Q24', 'post_Q25', 'post_Q27', 'post_Q28', 'post_Q30', 'post_Q31', 'post_Q33', 'post_Q34']

# Separating out the features
b = df.loc[:, features_all]
b = b.dropna()
x = b.loc[:, features].values
# Separating out the target
y = b.loc[:,['LGBTQ']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df[['LGBTQ']]], axis = 1)



import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['LGBTQ', '0_Not_LGBTQ']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['LGBTQ'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()





# Religion
# Separating out the target
y = b.loc[:,['Religion']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df[['Religion']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Abrahamic', '0_Other']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Religion'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()






# Gender
# Religion
# Separating out the target
y = b.loc[:,['Gender']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df[['Gender']]], axis = 1)



import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['0_Man', 'Not_Man']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Gender'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()
