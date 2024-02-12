import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Sequential
import nltk
from keras.preprocessing.text import Tokenizer

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
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    correlation_constant = data[row].corr(data[col])
                    correlation_dataset.loc[row, col] = correlation_constant
                except Exception as e:
                    pass
            else:
                continue

    return correlation_dataset

# Convert the categorial data of dataset into numeric data
def Categorial_to_num(data):
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            continue
        else:
            try:
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(data[col])

                data[col] = data[col].apply(lambda x: tokenizer.texts_to_sequences([x])[0][0] if x is not None else None)
            except Exception as e:
                print(e)

if __name__ == '__main__':
    df = pd.read_csv('Churn_Modelling.csv')
    print(df.head(5))
    rln = Correlation_in_dataset(df)
    print(rln)

