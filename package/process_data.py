import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from package.miscellaneous import nbits
from package.miscellaneous import export_params

def fill_tis(df):
    data = df.copy()
    data['rpr_dt'] = pd.to_datetime(data['rpr_dt'])
    data['wrty_start_dt'] = pd.to_datetime(data['wrty_start_dt'])
    negative_tis_i = data[data['tis'] < 0].index.tolist()
    data.loc[negative_tis_i,'tis'] = (data['rpr_dt'].loc[negative_tis_i] - data.loc[negative_tis_i,'wrty_start_dt']).dt.days/30

    #fill the remaining: when wrty_start_d or rpr_dt is nan 
    for mdl_yr in np.sort(data.mdl_yr.unique()):
        index_mdl_yr = data[data.mdl_yr == mdl_yr].index
        d_mdl_yr = data.loc[index_mdl_yr]
        index_nan_tis_yr = d_mdl_yr[d_mdl_yr['tis'].isna()].index
        
        if len(index_nan_tis_yr) > 0:
            d_mdl_yr.loc[index_nan_tis_yr, 'tis'] = d_mdl_yr.tis.mean()*np.ones(len(index_nan_tis_yr))
            data.loc[index_mdl_yr] = d_mdl_yr
        
    return data

def add_noise(X, y, w):

    y_noise = []

    X_noise = np.array(X).copy()

    shape = np.shape(y[0])
    np.random.seed(124)
    for i in range(0, len(y)):
        y_noise.append(np.array(y[i]) + np.random.normal(0.9, np.max(y[i])*w, shape))
    
    X_plus_noise = np.concatenate([X, X_noise], axis=0)

    return X_plus_noise, np.concatenate([y, y_noise], axis=0)

def shuffle(X, y):
    
    indexes = np.arange(len(y))
    np.random.shuffle(indexes) #randomiza os indices
    np.random.shuffle(indexes)
    np.random.shuffle(indexes)
    
    X_s = []
    y_s = []
    
    for i in indexes:
        X_s.append(X[i])
        y_s.append(y[i])
        
    return np.array(X_s), np.array(y_s)

def shuffle_to_cluster(X):
    
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    np.random.shuffle(indexes)
    np.random.shuffle(indexes)
    
    X_s = []
    
    for i in indexes:
        X_s.append(X[i])
        
    return X_s

def slice_df(df, params, filter_col, y_min, isnorm=True):
    """Slice to fit and validation data
    @f_df: 
    """
    fit_norm_df = []
    val_norm_df = []

    y_c = params['y_c']
    x_c = params['x_c']
    if isnorm:
        y_min = y_min / params['y_max']
        y_c = y_c / params['y_max']
        x_c = x_c / params['x_max']

    f_g_df = df.groupby(by=filter_col)

    #make sure that's exists enough data on y axis
    f_gmax_df = f_g_df.max()
    filters = f_gmax_df[f_gmax_df[params['y_col']] > y_min].index

    for filter_ in filters:

        df_filter = f_g_df.get_group(filter_)
        df_filter.loc[:,filter_col] = filter_

        fit_df = df_filter[df_filter[params['y_col']] <= y_c]
        fit_df = fit_df[fit_df[params['x_col']] <= x_c]

        val_df = df_filter[df_filter[params['y_col']] > y_c]
        val_df = val_df[val_df[params['x_col']] > x_c]

        fit_norm_df.append(fit_df)
        val_norm_df.append(val_df)

    return pd.concat(fit_norm_df), pd.concat(val_norm_df)

def build_database(df, filter_c_name, params, uri):
    """
    """
    df = df.copy()
    df = df[df[params['x_col']] <= params['x_k']]
    df = df[df[params['y_col']] <= params['y_k']]
    df_grouped = df.sort_values(by=[filter_c_name, params['x_col'], params['y_col']]).groupby(by=filter_c_name)

    database = []
    i = 0

    for filter_ in df_grouped.indices:
        database.append(df_grouped.get_group(filter_)['z'].tolist())
        i+=1

    export_params(uri, database)
    return database

def rm_descrease(arr, window_length):
    
    #to this fixed tis, we have len(y_df.index) curves z(odom). 
    for i in range(1,arr.size):
        arr[i] = max(arr[i-1], arr[i])

    #smooth the curve
    arr = savgol_filter(x = arr, 
                        window_length = window_length, 
                        polyorder = 3 )
    return arr

def post_processing(input_df, params, z_col, window_per=0.10):
    """
    Remove possible decrease and Smooth the surface. Must result of the NN is with too many brakings.
    @input_df
    @windows_per: length of the window as a percentage of the length of the arrays
    """
    index = input_df[input_df[z_col] < 0].index
    input_df.loc[index, z_col] = 0

    input_y_g_df = input_df.groupby(by=params['y_col'])

    x_window_len = int((params['x_max'] / params['x_step'] + 1)*window_per)
    y_window_len = int((params['y_max'] / params['y_step'] + 1)*window_per)
    ## To avoid dz/dx < 0
    for yi in input_y_g_df.indices.keys():
        yi_df = input_y_g_df.get_group(yi)
        input_df.loc[yi_df.index, z_col] = rm_descrease(np.array(yi_df[z_col]), y_window_len)

    input_x_g_df = input_df.groupby(by=params['x_col'])

    ## To avoid dz/dy < 0
    for xi in input_x_g_df.indices.keys():
        xi_df = input_x_g_df.get_group(xi)
        input_df.loc[xi_df.index, z_col] = rm_descrease(np.array(xi_df[z_col]), x_window_len)
        
    return input_df


def xy_acc_data(claim_df, prod_count = 1, params = None):

    #Build cumulative bivariate distribuition by elements that exists on dataset.
    count_claims_df = claim_df[[params['key'],params['x_col'],params['y_col']]].groupby(by=[params['x_col'],params['y_col']]).agg({params['key']: 'count'}).reset_index().rename(columns={params['key']: "count_key"})
    #count_claims_df.loc[:,"sum_count"] = count_claims_df['count_key'].cumsum()
    
    #All comibnations of x,y sorted and with step and max defined
    count_claims_df = pd.concat([pd.DataFrame([{params["x_col"]: -1, params["y_col"]: -1, "count_key": 0}]), count_claims_df])
    count_claims_arr = count_claims_df[[params['x_col'],params['y_col'],'count_key']].values
    xy_cartesian = np.array(np.meshgrid(np.arange(0, params['x_max']+params['x_step'], params['x_step']), np.arange(0, params['y_max']+params['y_step'], params['y_step']), indexing='ij')).T.reshape(-1, 2)
    acc_clm = []
    for xy in xy_cartesian: #Filter Iteration: assuming count_claims_arr is sorted by ascending
        f = count_claims_arr[:,0:2] <= np.reshape(xy, (2,)) #compare columns 0 and 1 = (x,y) with current x_i, y_i and return n of (Bool, Bool)
        acc_clm.append(np.sum(count_claims_arr[:,2][np.logical_and(f[:,0],f[:,1])]))
        #a[np.logical_and(f[:,0],f[:,1])][-1] = xy_acc_clm .: Do logical and to convert n of (Bool, Bool) to n of Bool and get all elements that match.
        """xy_acc_clm is is the accumulative sum of claims where x <= x_i nd y <= y_i. 
            The accumulative sum was performed before to improve performance. 
        """
    acc_clm_df = pd.DataFrame.from_records(xy_cartesian, columns = [params['x_col'], params['y_col']])
    acc_clm_df.loc[:,"z"] = 100*np.array(acc_clm) / prod_count
    return acc_clm_df


def build_data(clm_list, prod_list, cols_group, params):
    """Construct the data from history data to multiple different filters

    @clm_list: Pandas Data Frame with the claims listed in line. (DataFrame)
    @prod_list: ... production listed in line
    @cols_group: columns sorted to group the dataframe and get all possible filters. TODO: Must have the part number column as last column
    @id: unique identification of claim list row. Can be vin number or claim key, for example.
    @params: y_col: Must be time in service column name. 
                y_max: max to be tis considered.
                x_col: Must be mileage or kilometer, the odometer column.
                x_max: max odometer value considered
    """

    #Get all possible filters, by the defined columns to group
    clm_list.loc[:,"count"] = 1
    #clm_list.loc[:,params['x_col']] = (clm_list[params['x_col']] / params['x_step']).astype(np.int16)*params['x_step'] #discretize to x_step

    clm_grouped = clm_list[[params['key'], "count", params['x_col'], params['y_col']]+cols_group].groupby(by=cols_group)
    prod_grouped = prod_list[cols_group[:-1]+[params['key']]].groupby(by=cols_group[:-1]).count()

    clm_max_cols_df = clm_grouped.max()
    clm_max_cols_df = clm_max_cols_df[clm_max_cols_df[params['y_col']] > params['y_k']]
    filters = clm_max_cols_df[clm_max_cols_df[params['x_col']] > params['x_k']].index

    f_acc_dataset = []

    for filter_ in filters: #to each filter
        """
        Each iteraction will generate the accumulated dataframe to each filter.
        """
        clm_filter = clm_grouped.get_group(filter_)
        clm_filter = clm_filter.drop_duplicates(subset = params['key'], keep='first')
        
        p = params.copy()
        p['x_max'] = min(clm_filter[params['x_col']].max(), params['x_max']) #superior limit
        p['y_max'] = min(clm_filter[params['y_col']].max(), params['y_max'])

        xy_acc_df = xy_acc_data(claim_df=clm_filter.copy(),
                                prod_count = prod_grouped.loc[filter_[0:-1]][params['key']],
                                params=p
                                )
                        
        filter_str = str(filter_).replace("(","").replace("'", "").replace(",","_").replace(")", "").replace(" ", "").replace("/","")
        
        #if len(xy_acc_df.index) != 0:
        xy_acc_df.loc[:,'filter_'] = filter_str
        f_acc_dataset.append(xy_acc_df)
    
    return pd.concat(f_acc_dataset)

def norm_by_max(df, params, group=None, with_z=True):
    """Normalize the data
    @df: Dataset to be normalized
    @params: columns names with their max values. (Dict)  y_col: Must be time in service column name. 
                y_max: max to be tis considered.
                x_col: Must be mileage or kilometer, the odometer column.
                x_max: max odometer value considered
    @group: group name
    """

    norm_df = df.copy()
    norm_df.loc[:, params['x_col']] = norm_df[params['x_col']]/params['x_max'] #Odom column
    norm_df.loc[:, params['y_col']] = norm_df[params['y_col']]/params['y_max'] #tis column

    if with_z:
        if not group in params["z_max"].keys():
            params['z_max'][group] = float(norm_df.z.max())

        norm_df.loc[:,'z'] = norm_df.z/params['z_max'][group]
    
    return norm_df

def sel_dist_window(df, f_col, z_max, params):
    """Select distribution of the dataset by max of z. All filters which have z equals or greater than z_max will be removed.
    @df: accumulated dataframe
    @f_col: filter column
    @z_max: z value to cut the filters
    """
    new_df = []
    
    df_max_ag = df.groupby(by=[f_col]).max()#get max to each filter
    df_max_ag = df_max_ag[df_max_ag.z >= z_max[0]] #select only filters that have z max greater than z_max[0]
    df_max_ag = df_max_ag[df_max_ag.z <= z_max[1]]
    df_max_ag = df_max_ag[df_max_ag[params['x_col']] > params['x_k'] + 2] #select only filters which have enough data on x axis
    filters_1 = df_max_ag = df_max_ag[df_max_ag[params['y_col']] > params['y_k'] + 2].index #select only filters which have enough data on y axis

    df_g = df.groupby(by=[f_col]) #group again, without aggregate function

    zk_df = df[df[params['x_col']] == params['x_k']]
    zk_df = zk_df[zk_df[params['y_col']] == params['y_k'] ]
    zk_df = zk_df.groupby(by=[f_col]).max() #value of z at k position
    filters_2 = zk_df[zk_df['z'] > 0].index #remove all filters which haven't differentiation equals to zero on 0 -> k region

    filters = np.intersect1d(filters_1,filters_2)
    
    
    for filter_ in filters: #pass with only select filters
        new_df.append(df_g.get_group(filter_))
    
    print("Quantity of filters: ", len(new_df))
    return pd.concat(new_df)
    
def build_database(df, filter_c_name, params, uri):
    """Store the data to perform the verification task
    @df: accumulated dataset non normalized

    """
    
    filter_z_gdf = df.copy()[[filter_c_name,'z']].groupby(by=filter_c_name) #get dataframe grouped by filter to able the iterate by filter

    k_df = df[df[params['x_col']] <= params['x_k']].copy()
    k_df = k_df[k_df[params['y_col']] <= params['y_k']]
    k_df_grouped = k_df.sort_values(by=[filter_c_name, params['x_col'], params['y_col']]).groupby(by=filter_c_name)

    i = 0
    data_arr = {}
    z_max = {}
    for filter_ in k_df_grouped.indices.keys():
        data_arr[i] = k_df_grouped.get_group(filter_)['z'].tolist()
        z_max[i] = float(filter_z_gdf.get_group(filter_).z.max()) #z max over all values of the current filter
        i+=1

    database = {
        "z_max": z_max,
        "data_arr": data_arr
    }
    export_params(uri, database)
    return database
    
def norm_encode_bycluster(df, database, params, filter_c_name, norm_xy = True):
    """
    @database: Obs.: make sure the keys of database is sorted
    """
    df = df.copy()[[filter_c_name, params['x_col'], params['y_col'], "z"]]
    
    df_fk = df[df[params['x_col']] <= params['x_k']]
    df_fk = df_fk[df_fk[params['y_col']] <= params['y_k']]
    df_fk_grouped = df_fk.sort_values(by = [filter_c_name, params['x_col'], params['y_col']]).groupby(by=filter_c_name)

    df.loc[:,"group"] = 0 #just create new column with zeros to fill with the correct values after
    df.loc[:,"z_max"] = 0
    df_grouped = df.groupby(by=filter_c_name)

    data_arr = np.array(list(database['data_arr'].values()))
    
    df_max_grouped = df_grouped.max()
    df_max_grouped = df_max_grouped[df_max_grouped[params['x_col']] > params['x_k']]
    filters = df_max_grouped[df_max_grouped[params['y_col']] > params['y_k']].index #get only the filters which have enough data after cut
    
    if norm_xy:
        #x,y normalization:
        df.loc[:, params['x_col']] = df[params['x_col']] / params['x_max']
        df.loc[:, params['y_col']] = df[params['y_col']] / params['y_max']

    count = nbits(np.shape(data_arr)[0]) #number of columns to encode the group column in binary
    columns_g = np.arange(count) #count from 0 to max count. Names of columns of binary encoding

    z_maxdb = np.max(list(database['z_max'].values())) #overall z max

    norm_df = []
    for filter_ in filters:

        df_fk = df_fk_grouped.get_group(filter_)
        position = np.linalg.norm(data_arr-np.array(df_fk['z']), axis=-1).argmin(axis=0) #get the filter with the less distance 
        
        df_f = df_grouped.get_group(filter_).copy()
        df_f.loc[:, "group"] = position
        df_f.loc[:, "z"] = np.array(df_f["z"]) / database["z_max"][position]
        df_f.loc[:, "z_max"] = database["z_max"][position] / z_maxdb
        df_f.loc[:, filter_c_name] = filter_ 
        norm_df.append(df_f)

    norm_df = pd.concat(norm_df)
    b_arr = pd.DataFrame.from_records(np.unpackbits(np.array(norm_df['group']).astype(np.uint8).reshape((-1,1)), axis=1)[:,-count:], columns=columns_g)
    b_arr.index = norm_df.index

    return pd.concat([norm_df, b_arr], axis=1)


def create_X(params, data_arr, group:int):
    """
    data_arr = database['data_arr']
    """
    y_arr = np.arange(0, params['y_max']+params['y_step'], params['y_step'])
    x_arr = np.arange(0, params['x_max']+params['x_step'], params['x_step'])

    X_input = pd.DataFrame.from_records( data = np.array(np.meshgrid(x_arr, y_arr)).T.reshape(len(x_arr)*len(y_arr), 2),
                                      columns = [params['x_col'], params['y_col']])
              
    X_input.loc[:,"z_max"] = np.max(data_arr[int(group)]) / np.max(data_arr)
    group_arr = np.ones(len(X_input.index))*int(group)
    n_cols = nbits(np.shape(data_arr)[0])

    b_arr = pd.DataFrame.from_records(np.unpackbits(group_arr.astype(np.uint8).reshape((-1,1)), axis=1)[:,-n_cols:], columns=np.arange(n_cols))

    return pd.concat([X_input, b_arr], axis=1) #with columns: [x_col, y_col, z_max, n0, n1, .., nn]

def add_noise_by_filter(df, filter_col, per):

    df = df.copy()
    df_g = df.groupby(by=filter_col)
    df_meanz = df_g.mean()
    df_count = df_g.count()

    df_list = [df]

    for filter_ in df_g.indices.keys():
        df_filter_n = df_g.get_group(filter_).copy()
        std = df_meanz['z'][filter_]*per
        df_filter_n.loc[:,"z"] = df_filter_n["z"] + np.random.normal(std, std, df_count['z'][filter_]) #normal(mean, std, quantity)
        df_list.append(df_filter_n)

    return pd.concat(df_list)

def find_zero_dzdx_pos(matrix, dz_per):
    """To accumulated 2D z arrays, find the position which the differenciation stop being zero,
     going through 0 axis and from last position to first.

    Generaly,
    Find dz/dx -> diff on 0 axis
    Find max on the y direction -> will give an array with the max values on x direction. Once matrix have y of x array. max on axis 1
    Find position where the differenciation is different of zero from top to floor -> flip array and argmax on axis 0 plus 1  the differenciation
    """
    dz_dx = np.diff(matrix, axis=0) #the differenciation through x axis
    dz_dx_max_p = np.argmax(np.sum(np.diff(matrix, axis=0), axis=0)) #the position of the path with most high differenciation
    dz_dx_max_arr = np.append(dz_dx[:,dz_dx_max_p], 0) #finally the most high differenciation line through x axis. Add first element as default 0.
    inverted_x_p = (np.flip(dz_dx_max_arr) >= dz_dx_max_arr.max()*dz_per).argmax(axis=0)

    return np.shape(matrix)[0] - inverted_x_p -1 #get x length

def cut_surface_flattening(df, params, dz_per):
    """
    Cut the flattening of the surface.
    Columns with the shape = [x_col, y_col, z, ..., n_col]. z column must be at position 3.
    The dataframe must be sorted by filter, x_col, y_col
    """
    df = df.copy()

    #transforming the data in two dimensional shape
    x_p = int(df[params["x_col"]].max() / params['x_step'])+1 #quantity of positions through x axis
    y_p = int(df[params["y_col"]].max() / params['y_step'])+1 #quantity of positions through y axis
    
    z_2D = df.values[:,3].reshape((x_p, y_p)) #get z in two dimensional

    x_c = find_zero_dzdx_pos(z_2D, dz_per=dz_per) #go find where dz/dx stops to be 0, and dz/dy stop to be 0
    y_c = find_zero_dzdx_pos(z_2D.T, dz_per=dz_per)

    x_arr = df[params['x_col']].values / params['x_step'] #Get x array in term of positions. Can be obtained by cartesian product instead too.
    y_arr = df[params['y_col']].values / params['y_step']

    indices = (x_arr <= x_c) + (y_arr <= y_c) #cut using square

    """
    ra = (x_p-x_c)**2 #find readius of elipse
    rb = (y_p-y_c)**2
    indices_2 = (np.square(x_arr - x_c) / ra + np.square(y_arr - y_c) / rb) >= 1 #cut using circle
    indices = indices * indices_2
    """
    
    return df.iloc[indices]

def cut_flattening_filters(df, params, filter_c, dz_per):
    
    df = df[[filter_c, params['x_col'], params['y_col'], "z", ]].sort_values(by=[filter_c, params['x_col'], params['y_col']])
    df_g = df.groupby(by=filter_c)

    cutted_df = []
    for filter_ in df_g.indices.keys():
        cutted_df.append(cut_surface_flattening(df_g.get_group(filter_), params, dz_per))

    return pd.concat(cutted_df)