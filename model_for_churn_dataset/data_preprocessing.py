import pandas as pd
from Explortory_data_utils.Edv_utils import plot_kde_with_means
from Explortory_data_utils.Eda_utils import Categorial_to_num


df = pd.read_csv('model_for_churn_dataset\Churn_Modelling.csv')
Categorial_to_num(df)
# t_test(df, 'Exited', 'Balance', save=None, folder_name='model_for_churn_dataset\plots', file_name='t_test_b/w_exitednd_gender')
plot_kde_with_means(df, 'Geography', 'Balance')