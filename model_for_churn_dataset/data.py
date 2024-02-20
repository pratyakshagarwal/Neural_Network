import pandas as pd
from Explortory_data_utils.Eda_utils import Categorial_to_num
from Data_preprocessing_utils.Data_preprocessing_util import Scaling

def preprocess_data(data, columns):
    # Scaling the Required columns 
    data = Scaling(data, )

    # Converting the categorial data to numeric data
    Categorial_to_num(data)

    return data

