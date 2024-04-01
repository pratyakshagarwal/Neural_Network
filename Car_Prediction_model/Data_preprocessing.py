import pandas as pd
import numpy as np
from Data_preprocessing_util.Data_preprocessing_utils import Categorial_to_num, Scaling, binning_data
from sklearn.model_selection import train_test_split

df = pd.read_csv('Car_Prediction_model\car_price_prediction.csv')

def preprocess_data(df):
    df['Mileage'] = df['Mileage'].str.extract('(\d+)').astype(float)
    Categorial_to_num(df)

    df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')


    target_column = 'Price'
    input_columns = df.columns.difference([target_column])
    # print(input_columns)

    inputs_df = df[input_columns]
    target_df = df[target_column]

    return inputs_df, target_df

df = pd.read_csv('Car_Prediction_model\car_price_prediction.csv')

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



if __name__ == "__main__":
    print(X_train.head())
    # print(df.head(5))
    # inputs, target = preprocess_data(df)
    # print(inputs.dtypes)
    # print(inputs.shape)
    # print(target.head(5))
    # print(target.shape)