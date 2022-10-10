import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf

#package imports
from package.process_data import *
from package.cluster_hundler import *
from package.model_hundler import *
from package.miscellaneous import *

def get_conv_model(params):
    #x_o = int(params['x_max'] / params['x_step'] + 1)
    #y_o = int(params['y_max'] / params['y_step'] + 1)

    x_i = int(params['x_k'] / params['x_step'] + 1)
    y_i = int(params['y_k'] / params['y_step'] + 1)

    inputs = tf.keras.Input(shape=(x_i, y_i, 1))
    conv = tf.keras.layers.Conv2D(32, kernel_size=(2,2), strides=1, activation='relu', padding='same')(inputs)
    conv = tf.keras.layers.Conv2D(64, kernel_size=(2,2), strides=1, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.Conv2D(64, kernel_size=(2,2), strides=1, activation='relu', padding='same')(conv)
    batch = tf.keras.layers.BatchNormalization()(conv)

    expand = tf.keras.layers.Conv2DTranspose(filters = 32,
                                                padding = "valid",
                                                kernel_size = (4, 4),
                                                strides=(2,3),
                                                activation='relu')(batch)

    expand = tf.keras.layers.Conv2DTranspose(filters = 32,
                                                padding = "valid",
                                                kernel_size = (4, 4),
                                                strides=(1,1),
                                                activation='relu')(expand)

    expand = tf.keras.layers.Conv2DTranspose(filters = 32,
                                                padding = "valid",
                                                kernel_size = (1, 3),
                                                strides=1,
                                                activation='relu')(expand)

    norm = tf.keras.layers.BatchNormalization()(expand)
    conv = tf.keras.layers.Conv2D(32, kernel_size = (2,2), padding='same', activation='relu')(norm)
    conv = tf.keras.layers.Conv2D(1, kernel_size = (2,2), activation='linear')(conv)

    model = tf.keras.Model(inputs, conv)

    model.compile(loss=tf.losses.MeanAbsoluteError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=['mae','mse'])

    model.summary()
    return model


def build_data_to_generator(df, filter_c_name, params):
    """Generate the data to fit cluster. The shape of output is (length, 3)

    @df: Dataframe accumulated and not normalized. Must have "z" column
    @filter_c_name: Filter column name. Usually as "filter_"
    @params: dict with x column name (odometer), y column name (tis) and so on.
        params['length']: length of each sample. The result is a list of samples with this length
    """

    X = []
    y = []

    df_f = df[df[params['x_col']] <= params['x_k']]
    df_f = df_f[df_f[params['y_col']] <= params['y_k']]

    df_g = df.sort_values([filter_c_name, params['x_col'], params['y_col']]).groupby(by=filter_c_name)
    df_f_g = df_f.sort_values([filter_c_name, params['x_col'], params['y_col']]).groupby(by=filter_c_name)

    z_max = 0

    x_c = int(params['x_k'] / params['x_step'])+1
    y_c = int(params['y_k'] / params['y_step'])+1

    x_t = int(params['x_max'] / params['x_step'])+1
    y_t = int(params['y_max'] / params['y_step'])+1

    for filter_ in df_g.indices:
        
        #X:
        f_df = df_f_g.get_group(filter_)
        X.append(np.array(f_df['z']).reshape((x_c, y_c)))

        #y:
        y.append(np.array(df_g.get_group(filter_)['z']).reshape((x_t, y_t)))

        z_max = max(z_max, f_df['z'].max())

    params['z_max']['general'] = z_max

    return np.array(X) / z_max, np.array(y) / z_max


if __name__ == "__main__":

    ##Generate the second architecture of the regressor model
    params = dict({"x_col": "milge",
                    "x_max": 70_000,
                    "x_step": 2_000,
                    "y_col": "tis_wsd",
                    "key": "vin_cd",
                    "y_max": 70,
                    "y_step": 1,
                    "y_k": 21, #number of tis to be considered as input to cluster
                    "x_k": 30_000, #number of milge to be considered as input to cluster
                    "z_max": dict(), #group1: value1, group2: value2, group3: value3
                    "k": 4
                    })
    
    claims_df = pd.read_csv("./data/claims.tsv", sep='\t')
    prod_df = pd.read_csv("./data/prod.tsv", sep='\t')

    #TODO: The data used is already constructed and normalized.
    #Build History Data
    print("Built History Data")
    history_data = build_data(  clm_list=claims_df, 
                                prod_list = prod_df, 
                                cols_group = ["veh_line_cd","eng_cd","mdl_yr","prt_num_causl_base_cd"], 
                                params=params)

    print("Remove Outliers")
    history_data = sel_dist_window(df = history_data, #TODO: add insert noise function
                                    f_col = "filter_",
                                    z_max = [2,80], #z_max: base and top
                                    params = params,
                                    max_filters=500)

    
    X,y =  build_data_to_generator(df = history_data,
                                            filter_c_name = "filter_",
                                            params = params)

    X,y = add_noise(X, y, 0.03)

    print("shape of X: ")
    print(np.shape(X))

    print("shape of y: ")
    print(np.shape(y))

    export_params(cluster_uri="./models/base_model_params.json", params=params)
    ##Generate the model
    model = get_conv_model(params)

    history = fit_model(model,
                        X = X,
                        y = y,
                        batch_size = 16,
                        verbose = 2,
                        validation_data=None,
                        uri_model="./models/base_conv.h5",
                        patience = 5000,
                        epochs=50_000
            )
    
