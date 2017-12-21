import numpy as np
import pandas as pd
import sys


def read_csv(csv_file):
    """
    Converts a given csv file to a pandas DataFrame
    :param csv_file: a csv file
    :return: a pandas Dataframe with the csv file information
    """

    df = pd.read_csv(csv_file)
    return df


def apply_method(method, df, means, min_rssi, list_of_bs):
    """
    Creates a features matrix given a method name
    :param method: the method name
    :param df: the DataFrame to convert
    :param means: mean of train set, validation set and test set
    :param min_rssi: the minimum rssi value
    :param list_of_bs: the list of all existing bsid
    :return: a pandas DataFrame that corresponds to the features matrix
    """
    if method == 'lat_lng_dummies':
        df_feat = lat_lng_dummies(df, list_of_bs, means, min_rssi)
    else:
        if method == 'rssi':
            df_feat = rssis(df, list_of_bs)
        else:
            df_feat = df
            print('lol')

    return df_feat


def get_means(df):
    """
    Computes latitude and longtude mean
    :param df: a pandas DataFrame
    :return: a DataFrame with the computed means
    """
    return df[['bs_lat', 'bs_lng']].mean()


def compute_total_mean(train, val, test):
    """
    Computes the weighted mean for train set, validation set and test set depending on the size of each set
    :param train: train DataFrame
    :param val: validation DataFrame
    :param test: test DataFrame
    :return: the weighted mean
    """
    train_size = len(train)
    val_size = len(val)
    test_size = len(test)

    means_train = get_means(train)
    means_val = get_means(val)
    means_test = get_means(test)

    total = (means_train * train_size + means_val * val_size + means_test * test_size) / (
        train_size + val_size + test_size)
    return total


def get_min(df):
    """
    Returns the minimum value of rssi
    :param df: a DataFrame
    :return: the minimum rssi of the DataFrame
    """
    return df['rssi'].min()


def compute_total_min(train, val, test):
    total = min([get_min(train), get_min(val), get_min(test)])
    return total


# METHOD 1
def lat_lng_dummies(df, list_of_bs, means, min_rssi):
    """

    Affects the latitude * rssi of a given observation and the longitude * rssi on an
    other column on the corresponding base station.
    Affects the latitude * min_rssi (and longitude * min_rssi respectively) everywhere else

    :param df: a DataFrame
    :param list_of_bs: the list of all of the bsid
    :param means: the weighted mean
    :param min_rssi: the minimum rssi
    :return: a feature matrix
    """
    list_lats = [str(bs) + '_lat' for bs in list_of_bs]
    list_lngs = [str(bs) + '_lng' for bs in list_of_bs]
    list_columns = list_lats + list_lngs + ['did']

    min_rssi = min_rssi
    mean_lats = means['bs_lat'] * min_rssi
    mean_lngs = means['bs_lng'] * min_rssi

    df_mess_bs_group = df.groupby(['objid'], as_index=False)  # group data by message (objid)
    messages = np.unique(df['objid'])
    nb_mess = len(messages)

    df_feat = pd.DataFrame(index=np.arange(nb_mess), columns=list_columns)

    df_feat.loc[:, :len(list_of_bs)] = mean_lats
    df_feat.loc[:, len(list_of_bs):2 * len(list_of_bs)] = mean_lngs
    idx = 0

    for key, elmt in df_mess_bs_group:
        lats = df_mess_bs_group.get_group(key)['bs_lat'].values
        lngs = df_mess_bs_group.get_group(key)['bs_lng'].values
        df_feat.loc[idx, 'did'] = df_mess_bs_group.get_group(key)['did'].values[0]
        rssi = df_mess_bs_group.get_group(key)['rssi'].values
        for r, bsid in enumerate(df_mess_bs_group.get_group(key)['bsid'], 0):
            lat = str(bsid) + '_lat'
            lng = str(bsid) + '_lng'
            df_feat.loc[idx, lat] = lats[r] * rssi[r]
            df_feat.loc[idx, lng] = lngs[r] * rssi[r]
        idx = idx + 1
    return df_feat


# METHOD 2
def rssis(df, list_of_bs):
    """
    Affects the rssi_value of a given observation on the corresponding base station

    :param df: the DataFrame
    :param list_of_bs: the list of all of the bsid
    :return: a feature matrix
    """
    list_columns = [str(bs) + '_rssi' for bs in list_of_bs] + ['did']

    df_mess_bs_group = df.groupby(['objid'], as_index=False)  # group data by message (objid)
    messages = np.unique(df['objid'])
    nb_mess = len(messages)

    df_feat = pd.DataFrame(np.zeros((nb_mess, len(list_columns))), columns=list_columns)

    idx = 0

    for key, elmt in df_mess_bs_group:
        values = df_mess_bs_group.get_group(key)['rssi'].values
        df_feat.loc[idx, 'did'] = df_mess_bs_group.get_group(key)['did'].values[0]
        for r, bsid in enumerate(df_mess_bs_group.get_group(key)['bsid'], 0):
            rssi = str(bsid) + '_rssi'
            df_feat.loc[idx, rssi] = values[r]
        idx = idx + 1
    return df_feat


def save_csv(df, name):
    """
    Save a given DataFrame into a CSV file
    :param df: the DataFrame
    :param name: the name of the CSV file
    :return:
    """
    df.to_csv(name, sep=';', index=False)


def ground_truth_const(df_mess, pos):
    """
    Compute the groundtruth
    :param df_mess: the DataFrame
    :param pos: the labels
    :return: the ground truth
    """
    df_mess_pos = df_mess.copy()
    df_mess_pos[['lat', 'lng']] = pos

    ground_truth_lat = df_mess_pos.groupby(['objid']).mean()['lat']
    ground_truth_lng = df_mess_pos.groupby(['objid']).mean()['lng']

    frames = [ground_truth_lat, ground_truth_lng]
    ground_truth = pd.concat(frames, axis=1)

    return ground_truth


if __name__ == '__main__':
    # args : csv_train csv_val csv_test csv_train_y csv_val_y method
    args = sys.argv[1:]
    if len(args) < 6 or len(args) > 6:
        print("""\
        Usage: csv_train csv_val csv_test csv_train_y csv_val_y method \
        
        Available methods : lat_lng_dummies, rssi
        """)
        sys.exit(1)

    # convert to DataFrames
    train = read_csv(args[0])
    val = read_csv(args[1])
    test = read_csv(args[2])

    y_train = read_csv(args[3])
    y_val = read_csv(args[4])

    # determine all Base stations that received at least 1 message
    list_of_bs = np.union1d(np.union1d(np.unique(train['bsid']), np.unique(val['bsid'])), np.unique(test['bsid']))

    means = compute_total_mean(train, val, test)
    min_rssi = compute_total_min(train, val, test)

    l_train = len(train)
    l_val = len(val)
    l_test = len(test)
    print("Original lengths : ", l_train, l_val, l_test)

    gt_train = ground_truth_const(train, y_train)
    gt_val = ground_truth_const(val, y_val)

    save_csv(gt_train, 'ground_truth_train.csv')
    save_csv(gt_val, 'ground_truth_val.csv')
    print("Length ground truths : ", len(gt_train), len(gt_val))

    # get method
    method = args[5]
    df_train = apply_method(method, train, means, min_rssi, list_of_bs)
    df_val = apply_method(method, val, means, min_rssi, list_of_bs)
    df_test = apply_method(method, test, means, min_rssi, list_of_bs)

    l_train_formatted = len(df_train)
    l_val_formatted = len(df_val)
    l_test_formatted = len(df_test)
    print("New lengths : ", l_train_formatted, l_val_formatted, l_test_formatted)

    print("     | Columns length")
    print("     | ", len(df_train.columns.values), len(df_val.columns.values), len(df_test.columns.values))

    # save
    save_csv(df_train, 'train_formatted_' + method + '.csv')
    save_csv(df_val, 'val_formatted_' + method + '.csv')
    save_csv(df_test, 'test_formatted_' + method + '.csv')
