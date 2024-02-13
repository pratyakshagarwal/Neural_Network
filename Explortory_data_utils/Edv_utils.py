import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import statistics as stat

# comparing two columns of a dataframe through visulization
def visual_comparison_bw_colms(data, column1, column2, save=None, folder_name=None, file_name=None):
    plt.scatter(data[column1], data[column2])
    plt.xlabel(column1)
    plt.ylabel(column2)

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.show()


# compare every column of the dataframe from one another
def visual_comparison_on_dataset(data, save=None, file_name=None, folder_name=None):
    data.dropna(inplace=True) # remove null values first
    sns.pairplot(data)

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.show()


# compare the target col from two differnt columns you might have to change this according to your dataset
def plot(data, target_col, col1, col2, save=None, file_name=None, folder_name=None):
    df_Exited = data[data[target_col] == 1]
    df_notExited = data[data[target_col] == 0]

    df_Exited = df_Exited.sample(1000)
    df_notExited = df_notExited.sample(1000)

    plt.scatter(df_Exited[col1], df_Exited[col2], label='Exited')
    plt.scatter(df_notExited[col1], df_notExited[col2], label='Not Exited')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.show()



# Draw regression line on the plot takes two columns as an argument
def draw_Regression_line(data, x, y, save=None, folder_name=None, file_name=None):
    sns.regplot(data=data, x=x, y=y, line_kws={'color': 'black'}) # The regression line is built-in to the .regplot object
    sns.despine(top=True, right=True)
    
    # Calculate the regression line
    m, b, r, p, err = stats.linregress(data[x], data[y])

    # Add the formula, r squared, and p-value to the figure
    textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
    textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
    textstr += 'p  = ' + str(round(p, 2))
    plt.text(0.17, 0.80, textstr, fontsize=12, transform=plt.gcf().transFigure)

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.show()


# plot a jointplot b/w two columns and save it in the given folder
def plot_a_jointplot(data, x, y, kind, save=None, folder_name=None, file_name=None):
    if kind is not None:
        sns.jointplot(data=data, x=x, y=y, kind=kind)
    else :
        sns.jointplot(data=data, x=x, y=y)

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.show()


# plot a kde plot b/w two columns
def plot_kde_with_means(data, target_col, col2, save=None, folder_name=None, file_name=None):
    Exited = data[data[target_col] == 1][col2]
    Not_Exited = data[data[target_col] == 0][col2]

    # Plot KDEs for the two groups
    sns.kdeplot(data=data, x=col2, hue=target_col, fill=True)

    # Add vertical lines for means
    plt.axvline(x=Exited.mean(), color='orange', label='Exited Mean')
    plt.axvline(x=Not_Exited.mean(), color='blue', label='Not Exited Mean')

    if save is not None:
        file_path = f'{folder_name}/{file_name}'
        plt.savefig(file_path)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass
    