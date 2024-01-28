import pandas as pd
import numpy as np
import json
import tensorflow as tf

from package.miscellaneous import *
from package.model_hundler import *
from package.process_data import *

from multiprocessing import Process

def fit_submodel(base_uri, X_fit, y_fit, validation, history, filter_group):
    
    #X_fit, y_fit = shuffle(fitdf_filter[[params['x_col'],params['y_col']]].values, np.array(fitdf_filter['z']))
    #X_val, y_val = shuffle(valdf_filter[[params['x_col'],params['y_col']]].values, np.array(valdf_filter['z']))
    sub_model = tf.keras.models.load_model(base_uri)
    sub_model.trainable = True
    sub_model.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.Adam(),
                metrics=['mae'])

    history_f = fit_model( model = sub_model,
                            X = X_fit,
                            y = y_fit,
                            validation_data = validation,
                            verbose = 0,
                            mode = 'val',
                            patience = 30,
                            batch_size = 16,
                            epochs = 1_000,
                            uri_model = "./models/val_sub_models/"+filter_group[0]+".h5").history

    history.append(dict({ "filter_": filter_group[0], #store the filter
                            "group": filter_group[1],
                            "loss": float(min(history_f['loss']))}))

    with open("./models/val_sub_models/history.json","w") as f:
        f.write(json.dumps(history))
        
def start_filter(base_m_uri, fit_grp_norm, filter_, val_grp_norm, columns, processes, history):
    fitdf_filter = fit_grp_norm.get_group(filter_) #fit data
    group = float(fitdf_filter.iloc[0]['group'])
    
    valdf_filter = val_grp_norm.get_group(filter_) #validation data

    X_fit, y_fit = fitdf_filter[columns].values, np.array(fitdf_filter['z'])
    X_val, y_val = valdf_filter[columns].values, np.array(valdf_filter['z'])

    p = Process(target = fit_submodel, args = (base_m_uri, X_fit, y_fit, (X_val, y_val), history, (filter_, group),))
    processes.append(p)
    p.start()
    print("Process ", filter_, " initialized")

def fit_submodels(base_m_uri, params, fit_norm_df, val_norm_df, k_columns, shunk = 12):
    #fit the models to each filter
    fit_grp_norm = fit_norm_df.groupby(by=["filter_"])
    val_grp_norm = val_norm_df.groupby(by=['filter_'])

    history = [] #store the history of train and test

    filters = list(fit_grp_norm.indices.keys())

    columns = [params['x_col'], params['y_col'], 'z_max']+list(range(0,k_columns))

    processes = []

    #Start batch #0. Fill the processes with shunk filters.
    for filter_ in filters[0:shunk]: #shunk filters initialized
        start_filter(base_m_uri, fit_grp_norm, filter_, val_grp_norm, columns, processes, history)
        
    for filter_ in filters[shunk:]: #The processes already filled, add one by step
        processes.pop(0).join() #Wait for a process finish. This will give a vacancy to the next process
        start_filter(base_m_uri, fit_grp_norm, filter_, val_grp_norm, columns, processes, history)
    
    print("Wait the processes finish")
    for p in processes:
        p.join()

if __name__ == '__main__':

    ##---------------------Execute------------------------------------------------
    model_path = "sub1/"
    sub_path = "val_sub_models_15"
    
    clm_df = pd.read_csv("./data/claims.tsv", sep="\t")
    prod_df = pd.read_csv("./data/production.tsv", sep="\t")
    
    #Open the cluster model, base model and the params
    database = load_db("./models/"+model_path+"/database.json")
    params = dict(json.loads(open("./models/"+model_path+"/base_model_params.json","r").read())) #normalizer
    base_m_uri = "./models/"+model_path+"/base.h5"

    params = params.copy()
    params['x_c'] = 1_000
    params['y_c'] = 15

    export_params("./models/"+model_path+"/"+sub_path+"/model_params.json",params)

    current_data = pd.read_csv("./data/fit_val_data_submodels.csv")
    
    #current_data = cut_flattening_filters(df=current_data, params=params, filter_c='filter_', dz_per=0.25)
    #General Normalization
    current_gnorm_df = norm_encode_bycluster(df = current_data,
                                        database = database,
                                        params = params,
                                        filter_c_name = "filter_",
                                        norm_xy=True)

    print("columns: ", current_gnorm_df.columns)
    fit_norm_df, val_norm_df = slice_df(df = current_gnorm_df,
                                        params = params,
                                        filter_col = "filter_",
                                        y_min = params['y_c']+3,
                                        isnorm=True)

    fit_norm_df = add_noise_by_filter(df = fit_norm_df, filter_col = "filter_", per = 0.05)
    #backup: to run evaluation
    fit_norm_df.to_csv("./data/fit_norm_submodels_df.csv", index=False) 
    val_norm_df.to_csv("./data/val_norm_submodels_df.csv", index=False)

    print("Fit sub models")

    ##------ Fit SubModels -------------------------------------------
    fit_submodels(  base_m_uri= base_m_uri,
                    params = params, 
                    fit_norm_df = fit_norm_df, 
                    val_norm_df = val_norm_df,
                    k_columns = nbits(np.shape(list(database['data_arr'].values()))[0]),
                    shunk = 10)

