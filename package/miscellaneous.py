import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import json
from numpy import ceil, log, sort, array

def nbits(n_max):
    if n_max <= 1:
        return 1
    else:
        return int(ceil(log(n_max)/log(2)))

    
def plot_df(data_df, x_col, y_col, z_col, title, w = 20, h = 20, dpi = 60):

    figure(figsize=(w,h),dpi=dpi)
    ax = plt.axes(projection ='3d')

    ax.scatter(data_df[x_col], data_df[y_col], data_df[z_col], c = data_df[z_col])
    ax.set_ylabel(y_col)
    ax.set_xlabel(x_col)
    ax.set_zlabel(z_col)
    # syntax for plotting
    ax.set_title(title)

    plt.show()

def plot_surface(df, params, z_col="z", title="Surface", w=14, h=9):
    # Creating figure
    figure(figsize =(w, h))
    ax = plt.axes(projection ='3d')

    df_arr = df[[params['x_col'], params['y_col'], z_col]].sort_values(by=[params['x_col'], params['y_col']]).values
    x_p = int(params['x_max'] / params['x_step'])+1
    y_p = int(params['y_max'] / params['y_step'])+1

    z_2D = df_arr[:,2].reshape((x_p, y_p))
    x_2D = df_arr[:,0].reshape((x_p, y_p))
    y_2D = df_arr[:,1].reshape((x_p, y_p))
    
    ax.plot_surface(x_2D, y_2D, z_2D)
    ax.set_title(title)
    ax.set_ylabel(params['y_col'])
    ax.set_xlabel(params['x_col'])
    ax.set_zlabel(z_col)

    plt.show()

def create_path(path):

    step_path = ""
    subs = path.split("/")
    for p in subs[1:]:

        step_path = step_path + "/" + p
        p_i = subs[0] + "/" + step_path

        if not os.path.exists(p_i):
            os.mkdir(p_i)
            
def export_params(uri, params):
    
    create_path(os.path.split(uri)[0])
    with open(uri, "w") as f:
        f.write(json.dumps(params))

def sort_dict_keys(dictionary):
    dictionary = dictionary.copy()

    keys_sorted = sort(array(list(dictionary.keys())).astype(int)).tolist()
    new_dict = {}

    for key in keys_sorted:
        new_dict[key] = dictionary[str(key)]

    return new_dict

def load_db(uri):
    f = open(uri, "r")
    database = json.loads(f.read())
    f.close()
    
    database['z_max'] = sort_dict_keys(database['z_max'])
    database['data_arr'] = sort_dict_keys(database['data_arr'])
    return database
