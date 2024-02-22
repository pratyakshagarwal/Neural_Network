import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Sequential
import nltk
from keras.preprocessing.text import Tokenizer
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.formula.api import ols
import statistics as stats

# take a pandas func as an argument return after applying it to every columns
def EdaFunc(data, evaluation):
    for column in data.columns:
        try :
            print("{} for {} is {}".format(evaluation, column, getattr(data[column], evaluation)()))
        except Exception as e:
            print(e)

# take a pandas method as an argument return after applying it to every columns
def EdaWFunc(data, evaluation):
    for column in data.columns:
        try :
            print("{} for {} is {}".format(evaluation, column, getattr(data[column], evaluation)))
        except Exception as e:
            print(e)

# takes one column as argment and return its correlation with other column of the dataset
def find_Correlation_bw_colms(data, target_column):
    for column in data.columns:
        if column == target_column:
            continue
        else:
            try:
                print('the correlation b/w {} and {} is {}:'.format(column, target_column, getattr(data[target_column], 'corr')(data[column])))
            except Exception as e:
                pass


# return a dataset with correlations b/w the columns 
def Correlation_in_dataset(data):
    correlation_dataset = pd.DataFrame(columns=data.columns, index=data.columns)
    
    for row in data.columns:
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    correlation_constant = data[row].corr(data[col])
                    correlation_dataset.loc[row, col] = correlation_constant
                except Exception as e:
                    pass
            else:
                continue

    return correlation_dataset


# measure Heteroscedasticity through statsmodels take a formula as an argument which is nothing more than a two columns
def Heteroscedasticity(data, formula):
    # Fit the OLS model
    model = ols(formula=formula, data=data).fit()

    white_test = het_white(model.resid,  model.model.exog)
    bp_test = het_breuschpagan(model.resid, model.model.exog)
  
    output_df = pd.DataFrame(columns=['LM stat', 'LM p-value', 'F-stat', 'F p-value'])
    output_df.loc['White'] = white_test
    output_df.loc['Breusch-Pagan'] = bp_test

    output_df.round(3)
    return output_df

# This fucntion take two colmuns of a dataset as an argument and implies t-test on them 
def independent_t_test(feature, label, alpha=0.05):
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(feature, label)

    # Check the results
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the groups.")


# This fucntion take two colmuns of a dataset as an argument and implies anova on them you might want to change this func according to your data
def anova(data, feature, label):
    groups = data[feature].unique() # discover each unique group value
    grouped_values = []           # create an overall list of keep track of the label sub-lists
    for group in groups:          # for each unique group value
        grouped_values.append(data[data[feature]==group][label])  # append a sub-list of label values into the overall list
    return stats.f_oneway(*grouped_values)  


if __name__ == '__main__':
    pass

