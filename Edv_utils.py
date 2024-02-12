import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# comparing two columns of a dataframe through visulization
def visual_comparison_bw_colms(data, column1, column2, folder_name, file_name):
    plt.scatter(data[column1], data[column2])
    plt.xlabel(column1)
    plt.ylabel(column2)

    file_path = f'{folder_name}/{file_name}'
    plt.savefig(file_path)

    plt.show()


# compare every column of the dataframe from one another
def visual_comparison_on_dataset(data, file_name, folder_name):
    data.dropna(inplace=True) # remove null values first
    sns.pairplot(df)

    file_path = f'{folder_name}/{file_name}'
    plt.savefig(file_path)

    plt.show()


# compare the target col from two differnt columns you might have to change this according to your dataset
def plot(data, target_col, col1, col2, file_name, folder_name):
    df_Exited = data[data[target_col] == 1]
    df_notExited = data[data[target_col] == 0]

    df_Exited = df_Exited.sample(1000)
    df_notExited = df_notExited.sample(1000)

    plt.scatter(df_Exited[col1], df_Exited[col2], label='Exited')
    plt.scatter(df_notExited[col1], df_notExited[col2], label='Not Exited')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()

    file_path = f'{folder_name}/{file_name}'
    plt.savefig(file_path)
    plt.show()


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



# Draw regression line on the plot takes two columns as an argument
def draw_Regression_line(data, x, y, folder_name, file_name):
    sns.regplot(data=data, x=x, y=y, line_kws={'color': 'black'}) # The regression line is built-in to the .regplot object
    sns.despine(top=True, right=True)
    
    # Calculate the regression line
    m, b, r, p, err = stats.linregress(df[x], df[y])

    # Add the formula, r squared, and p-value to the figure
    textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
    textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
    textstr += 'p  = ' + str(round(p, 2))
    plt.text(0.17, 0.80, textstr, fontsize=12, transform=plt.gcf().transFigure)

    file_path = f'{folder_name}/{file_name}'
    plt.savefig(file_path)
    plt.show()


# plot a jointplot b/w two columns and save it in the given folder
def plot_a_jointplot(data, x, y, kind, folder_name, file_name):
    if kind is not None:
        sns.jointplot(data=data, x=x, y=y, kind=kind)
    else :
        sns.jointplot(data=df, x=x, y=y)

    file_path = f'{folder_name}/{file_name}'
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('Churn_Modelling.csv')
    print(df.head(5))
    output_df = Heteroscedasticity(df, 'Balance~Age')
    print(output_df)
    # draw_Regression_line(df, 'Age', 'Balance')
    # sns.jointplot(data=df, x='Age', y='Balance')
    plot_a_jointplot(df, 'Age', 'Balance', "hex", 'plots', 'jointplot_bw_agend_balance')