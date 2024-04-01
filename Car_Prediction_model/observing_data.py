# import necessary library
import time
import pandas as pd
import numpy as np
from Explortory_data_utils.Eda_utils import EdaFunc, EdaWFunc, Correlation_in_dataset, check_null_data
from Explortory_data_utils.Edv_utils import visual_comparison_bw_colms
from Data_preprocessing_util.Data_preprocessing_utils import Categorial_to_num

if __name__ == "__main__":
    df = pd.read_csv('Car_Prediction_model\car_price_prediction.csv')

    folderpath = 'Car_Prediction_model\plots'

    print(df.head(5))

    EdaFunc(df, 'describe')
    EdaFunc(df, 'info')
    EdaWFunc(df, 'dtype')
    EdaWFunc(df, 'shape')

    Categorial_to_num(df)
    print(df.head(5))

    print(Correlation_in_dataset(df))

    for column in df.columns:
        if column == 'price': 
            continue
        visual_comparison_bw_colms(df, column, 'Price', save=None, folder_name=folderpath, file_name=f'{column}vsprice.png')
        time.sleep(2)



'''
Through This observation of data i noticed some key features:

1. There are 8 objects datatype columns that needs to be conveted into numerical data
2. while checking graphs and correlation dataframe. I noticed that there is not significant relation ship between columns and the target(price)
3. We need to handle Mileage column sepreately because of it nither being numerical column not object
4. Ther is no blank row or blank data in a particular data
'''


