# cd Documents/AnBeh_Revised/SurveyResults

library(MASS)
df = read.csv('data_Nov22.csv')

list = c('Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14', 'Q16', 'Q17', 'Q18', 'Q19', 'Q21', 'Q22', 'Q24', 'Q25', 'Q27', 'Q28', 'Q30', 'Q31', 'Q33', 'Q34', 'Q58', 'Q37_total', 'Q38_total')


models <- lapply(list, function(x) {
    model = lm(formula = paste0("post_", x, " ~ pre_", x, "+ Course + LGBTQ + Gender + Religion + Course:LGBTQ + Course:Gender + Course:Religion"), data=df)
    AIC = stepAIC(model)
    AIC$call
})

models
