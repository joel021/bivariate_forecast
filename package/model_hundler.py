import numpy as np
import tensorflow as tf
from package.process_data import post_processing, norm_by_max, create_X
from pandas import DataFrame
from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture
import joblib
import numpy as np

def fit_model(model, X, y, validation_data, uri_model, verbose=2, batch_size=8, mode = 'fit', patience = 1000, epochs=20_000):

    monitor = 'mae'
    if mode == 'val':
        monitor = "val_mae"

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                patience=patience,
                                                mode='min')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(uri_model, 
                                                    monitor=monitor, 
                                                    save_best_only=True,
                                                    mode='min',
                                                    verbose=verbose
                                                    )
    #model.trainable = True
    return model.fit(x = np.array(X),
                y = np.array(y),
                validation_data = validation_data,
                epochs= epochs,
                shuffle = True,
                batch_size=batch_size,
                verbose=verbose, #0 -> silent, 1 -> more verbosily, 2 -> one line
                callbacks=[early_stopping, checkpoint],
                )

def regression_model(n_i):
    #curve adjustment model to this mileage
    
    inputs = tf.keras.Input(shape=(n_i,)) #input = [x,y,c0,c1,..,cn]

    dense = tf.keras.layers.Dense(36, activation='relu',
                                    #kernel_regularizer=tf.keras.regularizers.l2()
                                    )(inputs)
    norm = tf.keras.layers.Normalization()(dense)

    dense = tf.keras.layers.Dense(64, 
                                    activation='relu', 
                                    #kernel_regularizer=tf.keras.regularizers.l2()
                                    )(norm)
    norm = tf.keras.layers.Normalization()(dense)
    
    dense = tf.keras.layers.Dense(128,
                                    #kernel_regularizer=tf.keras.regularizers.l2(), 
                                    activation="relu")(norm)

    dense = tf.keras.layers.Dense(84, 
                                    activation="relu")(dense)       

    #norm = tf.keras.layers.Normalization()(dense)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(1, activation='linear')(dropout)

    model = tf.keras.Model(inputs=inputs, outputs=dense)

    model.compile(loss=tf.losses.MeanAbsoluteError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=['mae'])#'mse',

    return model

def predict_regression(model, database, params, group):

    X_input = create_X(params, list(database['data_arr'].values()), group)
    X_input.loc[:,"z_pred"] = model.predict(norm_by_max(df=X_input, params=params, with_z=False))

    X_input = post_processing( input_df=X_input,
                                params = params,
                                z_col = 'z_pred')

    X_input.loc[:, "z_pred"] = X_input['z_pred']*database['z_max'][group]
    
    return X_input


#--------------- CLUSTER HANDLER ----------------------------
def encoder(X=None, n_y = 16, n_x = 22, k = 2, cluster_uri="", patience = 1000, epochs=20_000):

    inputs = tf.keras.Input(shape=(n_y, n_x, 1)) #(n_x, ny, 1) = (h, w, 1) = (lines, rows, channels)

#------------- ENCONDER MODEL ------------------------------------
    conv1 = tf.keras.layers.Conv2D(12, kernel_size = 2)(inputs)
    norm1 = tf.keras.layers.LayerNormalization()(conv1)
    a1 = tf.keras.activations.relu(norm1)

    conv2 = tf.keras.layers.Conv1D(24, kernel_size = 3)(a1)
    norm2 = tf.keras.layers.LayerNormalization()(conv2)
    a2 = tf.keras.activations.relu(norm2)

    pool1 = tf.keras.layers.MaxPool2D()(a2)

    conv3 = tf.keras.layers.Conv1D(32, kernel_size = 3)(pool1)
    norm3 = tf.keras.layers.LayerNormalization()(conv3)
    a3 = tf.keras.activations.relu(norm3)

    pool2 = tf.keras.layers.MaxPool2D()(a3)

    conv4 = tf.keras.layers.Conv2D(64, kernel_size = 3)(pool2)
    norm4 = tf.keras.layers.LayerNormalization()(conv4)
    a4 = tf.keras.activations.relu(norm4)

    flatten = tf.keras.layers.Flatten()(a4)
    cluster = tf.keras.layers.Dense(k, activation = "softmax")(flatten)

    encoder = tf.keras.Model(inputs, cluster)
    #encoder.summary()

#--------- DECODER MODEL ------------------------------------------
    reshape = tf.keras.layers.Reshape((1,1,k))(cluster)
    tconv = tf.keras.layers.Conv2DTranspose(filters = 64,
                                            kernel_size=3,
                                            padding='valid',
                                            strides=1,
                                            activation='relu')(reshape)

    tconv = tf.keras.layers.Conv2DTranspose(filters = 32,
                                            padding = "valid",
                                            kernel_size = 4,
                                            strides=1,
                                            activation='relu')(tconv)

    tconv = tf.keras.layers.Conv2DTranspose(filters = 24,
                                            padding = "valid",
                                            kernel_size = 5,
                                            strides=1,
                                            activation='relu')(tconv)

    tconv = tf.keras.layers.Conv2DTranspose(filters = 12,
                                            padding = "valid",
                                            kernel_size = 6,
                                            strides=1,
                                            activation='relu')(tconv)

    tconv = tf.keras.layers.Conv2DTranspose(filters = 1,
                                            padding = "valid",
                                            kernel_size = (2, 8),
                                            strides=1,
                                            activation='relu')(tconv)

                                            
    enc_decoder = tf.keras.Model(inputs = inputs, outputs=tconv)
    

    encoder.compile(loss=tf.losses.MeanAbsoluteError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=['mae'])

    enc_decoder.compile(loss=tf.losses.MeanAbsoluteError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=['mae'])

    history = fit_model(model = enc_decoder,
                        X = X,
                        y = X,
                        validation_data = None,
                        uri_model = cluster_uri+".enc.dec.h5",
                        verbose=0, 
                        batch_size=16, 
                        mode = 'fit',

                        )
    encoder.save(cluster_uri)
    
    return history.history, encoder

def build_cluster(n_clusters, dataset_fit, cluster_uri):
    """ Fit and save the cluster model
    @n_cluster: number of clusters (int)
    @dataset_fit: normalized dataset
    """
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset_fit)
    #cluster_model = GaussianMixture(n_components=n_clusters, random_state=0).fit(dataset_fit)
    """
    Next work: Fit with best cluster and search how can two filter will be in the same cluster.
    The goal with cluster is mensure the distance between the current data the ours already knows surfaces to select one of they.
    The most closest model will be adjusted to the current data.
    """
    #save model
    joblib.dump(cluster_model, cluster_uri)
    return cluster_model

def pred_df_NNcluster(encoder, params, norm_df): #TODO define function to select best threshold by a metric
    """Make prediction to each filter using Neural Network model
    @cluster_model
    @norm_df: general normalized dataframe
    """
    norm_df_grouped = norm_df.groupby(by="filter_")
    classes = np.arange(0, params["k"])

    x_p = int(params['x_k'] / params['x_step'])+1
    y_p = int(params['y_k'] / params['y_step'])+1

    x_cut = params['x_k'] / params['x_max']
    y_cut = params['y_k'] / params['y_max']

    for filter_ in norm_df_grouped.indices:
        norm_f_df = norm_df_grouped.get_group(filter_)[[params['x_col'], params['y_col'], 'z']].sort_values(by=[params['x_col'], params['y_col']])
        #convert to shape expected by the model

        norm_f_arr = norm_f_df[norm_f_df[params['x_col']] <= x_cut]
        norm_f_arr = norm_f_arr[norm_f_arr[params['y_col']] <= y_cut]

        z_arr = np.array(norm_f_arr['z']).reshape((1, x_p, y_p, 1))

        pred_arr = encoder.predict(z_arr).argmax(axis=0)

        norm_df.loc[norm_f_df.index, "group"] = classes[pred_arr][0]

    norm_df.loc[:, "group"] = norm_df['group'].astype(int)
    return norm_df #predicted data

def pred_df_NNcluster(encoder_model, params, norm_df, filter_c):
    """Make prediction to each filter
    @cluster_model
    @norm_df: general normalized dataframe
    """
    norm_X_df = norm_df[norm_df[params['x_col']] <= params['x_k']]
    norm_X_df = norm_X_df[norm_X_df[params['y_col']] <= params['y_k']]
    df_grouped = norm_X_df.sort_values(by=[filter_c, params['x_col'], params['y_col']]).groupby(by=filter_c)[['z']]

    x_k = int(params['x_k'] / params['x_step'])+1
    y_k = int(params['y_k'] / params['y_step'])+1

    rows = int(params['x_max'] / params['x_step'] + 1)*int(params['y_max'] / params['y_step'] + 1)

    norm_pred_df = []
    enc_arr = []
    params['encoding'] = dict()

    for filter_ in df_grouped.indices:
        enc_arr_i = encoder_model.predict(df_grouped.get_group(filter_)['z'].values.reshape(1, x_k, y_k, 1))
        enc_arr.append(np.repeat(enc_arr_i, repeats=rows, axis=0)) #append an array with shape (rows, k)
        params['encoding'][filter_]
    enc_arr = np.array(enc_arr).reshape(len(df_grouped.indices)*rows, params['k'])

    norm_pred_df = DataFrame.from_records(np.array(norm_pred_df).reshape(len(norm_df.index), len(norm_df.columns)), columns=norm_df.columns)
    
    return norm_pred_df #predicted data
    
def pred_df_cluster(cluster_model, params, norm_df):
    """Make prediction to each filter
    @cluster_model
    @norm_df: general normalized dataframe
    """
    x_k = params['x_k'] / params['x_max']
    y_k = params['y_k'] / params['y_max']

    norm_pred_df = []

    norm_df.loc[:,"group"] = 0

    norm_df_grouped = norm_df.groupby(by="filter_")
    for filter_ in norm_df_grouped.indices:
        norm_f_df = norm_df_grouped.get_group(filter_)

        norm_fl_df = norm_f_df[norm_f_df[params['x_col']] == x_k]
        norm_fl_df = norm_fl_df[norm_fl_df[params['y_col']] <= y_k]

        norm_f_df.loc[:,"group"] = cluster_model.predict(np.array(norm_fl_df['z']).reshape(1,-1))[0]

        norm_pred_df.append(norm_f_df)
        
    norm_pred_df = DataFrame.from_records(np.array(norm_pred_df).reshape(len(norm_df.index), len(norm_df.columns)), columns=norm_df.columns)
    
    return norm_pred_df #predicted data
    