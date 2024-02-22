import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

# print the detail about the columns with missing values 
def check_null_data(data):
    missing_data = data.isnull()
    for column in missing_data.columns.values.tolist():
        # print(column)
        print (missing_data[column].value_counts())
        print("")  


# it handles the columns with missing values by replacing them with mean of the column using simpleimputer of sklearn
def handling_missing_value_through_mean(data, columns):
    missing_values = data.isnull()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(data[columns])
    data[columns] = imputer.transform(data[columns])
    return data


# it handles the columns with missing values by replacing them with mean of the column using pandas function
def handling_missing_value_through_mean_wthpd(data, columns):
    for column in columns:
        avg_loss = data[column].astype('float').mean(axis=0)
        data[column].replace(np.nan, avg_loss,inplace = True)
    
    return data


# it handles the columns with missing values by replacing them with frequency of the column using simple simpleimputer of sklearn
def handling_missing_value_through_frequency(data, columns):
    missing_values = data.isnull()
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(data[columns])
    data[columns] = imputer.transform(data[columns])
    return data


# it handles the columns with missing values by replacing them with frequency of the column using pandas function
def handling_missing_value_through_frequency_wthpd(data, columns):
    for column in columns:
        data[column].value_counts()
        maxval = data[column].value_counts().idxmax()
        data[column].replace(np.nan, maxval, inplace=True)   
    return data


# this func deletes the rows with missing value
def deleting_rows_with_missing_values(data, columns):
    data.dropna(subset=columns, axis=0, inplace=True)

    # reset index, because we droped two rows
    data.reset_index(drop=True, inplace=True)


# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
# and this fucntion process that for you
def binning_data(data, columns, group_names):
    for column in columns:
        bins = np.linspace(min(data[column]), max(data[column]), len(group_names))
        data[column] = pd.cut(data[column], bins, labels=group_names, include_lowest=True)
    return data


# spilt train and test data 
def Spiliting_in_test_nd_train(data):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1] ,test_size = 0.2,random_state = 90)
    return X_train, y_train, X_test, y_test


# scale the data to to whatever scale you want
def Scaling(data, columns, scaler=None):
    for column in columns:
        sc = StandardScaler()
        # Reshape the column to a 2D array
        column_data = data[column].values.reshape(-1, 1)
        # Fit and transform the data
        data[column] = sc.fit_transform(column_data)
    return data


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
    pass