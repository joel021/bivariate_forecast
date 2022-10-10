import pandas as pd
import numpy as np
from glob import glob
from scipy.special import erf
from scipy.optimize import curve_fit
import json
from sklearn.metrics import mean_absolute_error
from package.miscellaneous import export_params, load_db, plot_df, nbits
from package.process_data import norm_encode_bycluster
import warnings
warnings.filterwarnings("ignore")

class CommonDistribution():

    def __init__(self):
        pass
    
    def logo_normal(self, x, mi, sigma):
        return np.exp(-(np.log(x)-mi)**2/(2*sigma**2))/(x*np.sqrt(2*np.pi*sigma))

    def CD_logo_normal(self, x, mi, sigma):
        return 0.5*(1 + erf( (np.log(x) - mi) / (sigma*np.sqrt(2)) ) )

    def normal(self, x, mi, sigma):
        return np.exp(-(x-mi)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
        
    def weibull(self, x, lambda_, k):
        x = x.copy() #make a copy of x to not change the original and discart the original

        i_greater = x >= 0 #Case 1: indices where the elements of x is greater than or equals to 0
        i_non = x < 0 #Case 2: indices where the elements of x is less than 0

        x[i_non] = x[i_non]*0 #result to the Case 1
        
        x_g = x[i_greater]
        return ((k/lambda_)*(x_g/lambda_)**(k-1))*np.exp(-(x_g/lambda_)**k)

    def CD_weibull(self, x, lambda_, k):

        x = x.copy() #make a copy of x to not change the original and discart the original

        i_greater = x >= 0 #Case 1: indices where the elements of x is greater than or equals to 0
        i_non = x < 0 #Case 2: indices where the elements of x is less than 0

        x[i_non] = x[i_non]*0 #result to the Case 1
        x[i_greater] = 1 - np.exp(-(x[i_greater]/lambda_)**k) #result to the Case 2

        return x

    def accumulative(self, model, x_max):
        y_arr = []
        x_arr = np.arange(0.01, x_max+0.06, 0.06)
        sum_ = 0

        for x in x_arr:
            sum_ += model.density_func(x)

            y_arr.append(sum_)

        return np.array([self.min_max_scaler(x_arr), self.min_max_scaler(y_arr)])
    
    def min_max_scaler(self, arr):
        min_ = np.min(arr)
        max_ = np.max(arr)
        return (arr - min_)/(max_-min_)

    def bi_accumulative(self, fx, fy, x_max, y_max):
        
        fx_arr = self.accumulative(fx, x_max)
        fy_arr = self.accumulative(fy, y_max)

        range_y = range(len(fy_arr[0]))

        data_df = []
        for i in range(len(fx_arr[1])):

            for j in range_y:
                data_df.append({"x": fx_arr[0][i], "y": fy_arr[0][j], "z": min(fx_arr[1][i], fy_arr[1][j])})
        
        data_df = pd.DataFrame(data_df)
        data_df['x'] = self.min_max_scaler(data_df['x'])
        data_df['y'] = self.min_max_scaler(data_df['y'])
        data_df['z'] = self.min_max_scaler(data_df['z'])

        return data_df


def xmodel_prediction(func, model_args, params):
    """This function return the z values predicted to 0 <= x <= x_max and 0 <= y <= y_max
    
    """
    dataframe = []

    y_arr = np.arange(0, params['y_max'] + params['y_step'], params['y_step']) #get unique values to y
    ones = np.ones(len(y_arr)) #aux arr to make broadcast

    for x in model_args.keys(): #to each unique value of x
        df_i = pd.DataFrame({params['y_col']: y_arr,
                            params['x_col']: ones*x, #broadcast
                            "z": func(y_arr/params['y_max'], *model_args[x])})
        dataframe.append(df_i)

    dataframe_df = pd.concat(dataframe)
    dataframe_df = dataframe_df.sort_values(by=[params['x_col'], params['y_col']])

    return dataframe_df
    
def ymodel_prediction(func, model_args, params):
    """This function return the z values predicted to 0 <= x <= x_max and 0 <= y <= y_max
    """
    dataframe = []

    x_arr = np.arange(0, params['x_max'] + params['x_step'], params['x_step']) #get unique values to x
    ones = np.ones(len(x_arr)) #aux arr to make broadcast

    for y in model_args.keys(): #to each unique value of x
        df_i = pd.DataFrame({params['x_col']: x_arr,
                            params['y_col']: ones*y, #broadcast
                            "z": func(x_arr/params['y_max'], *model_args[y])})
        dataframe.append(df_i)

    dataframe_df = pd.concat(dataframe)
    dataframe_df = dataframe_df.sort_values(by=[params['x_col'], params['y_col']])

    return dataframe_df

def build_models(model_func, fit_norm_df, filter_col, params):
    """
    Return a dict of models. One model to each filter.
    @model_func: model to be fitted on the fit_df. Must common is accumulated Logo Normal and Weibull. (CommonDistribution.<function name>)
    @fit_df: data to fit (Pandas). OBS.: The x and y columns must not be normalized
    @filter_col: column which identify a unique filter
    """
    acc_models = {"func": model_func.__name__,  #store the function name
                  "params": params} #store the parameters
    
    popt = np.array([1,1]) #initialize the params

    g_df = fit_norm_df.groupby(by=filter_col)

    for filter_ in g_df.indices:
        
        df_filter = g_df.get_group(filter_)
        acc_models[filter_] = {}
        #x direction

        for x in df_filter[params['x_col']].unique():
            xy_df = df_filter[df_filter[params['x_col']] == x]
            try:
                popt, pcov = curve_fit(f = model_func,
                                        xdata = xy_df[params['y_col']] / params['y_max'], #norm y array before
                                        ydata = xy_df['z'], #z must be normalized
                                        #bounds = (-100,1)
                                        )
            except:
                pass
            acc_models[filter_][x] = popt
            acc_models[filter_]['group'] = int(df_filter.iloc[0]['group'])

    return acc_models