import pandas as pd
from Data_preprocessing_util.Data_preprocessing_utils import Scaling,Categorial_to_num 

def preprocess_data(data, columns):
    # Scaling the Required columns 
    data = Scaling(data, columns)

    # Converting the categorial data to numeric data
    Categorial_to_num(data)

    return data

if __name__ == '__main__':
    df = pd.read_csv('model_for_churn_dataset\Churn_Modelling.csv')
    data = preprocess_data(df, ['Balance', 'EstimatedSalary'])
    print(data.head(5))

# source venv/bin/activate
