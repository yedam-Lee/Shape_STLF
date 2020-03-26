import numpy as np
import pandas as pd
import collections

from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans

from utils.utils import get_array, get_label_df, get_transition_matrix, plot_trans_mtx, plot_cluster


def make_sequence(k, model, dist_type, df_test, label_df_train, label_df_test):
    arr_test_full, arr_test_norm = get_array(df_test, 24)

    for i in range(len(arr_test_full)):
        distance = []
        for cn in range(k):
            if dist_type == 'ed':
                dist = np.linalg.norm(arr_test_full[i] - model.cluster_centers_[cn])
            else:
                dist = dtw.distance(arr_test_full[i], model.cluster_centers_[cn])
            distance.append(dist)
        cluster = np.argmin(distance)
        label_df_test['label'][i] = cluster
    cluster_sequence = pd.concat([label_df_train, label_df_test], axis=0)
    cluster_sequence.sort_index(inplace=True)

    return cluster_sequence


def seperate_dataset(k, label_df, train):
    data_in_clusters = [[] for i in range(k)]
    for i in range(len(train)):
        label = label_df.loc[str(train.iloc[i].name.date())]
        data_in_clusters[label['label']].append(train.iloc[i])

    result = [[] for i in range(k)]
    for i in range(k):
        con_ = pd.concat(data_in_clusters[i], axis=1).T
        result[i].append(con_)

    return result


class Clusterer:

    def __init__(self, k):
        self.k = k
        self.model = TimeSeriesKMeans(n_clusters=k,
                                      n_init=2,
                                      metric="dtw",
                                      verbose=False,
                                      max_iter_barycenter=10,
                                      random_state=0)

    def fit(self, train_test_clu):
        df_train, df_test = train_test_clu[0], train_test_clu[1]

        arr_train_full, arr_train_norm = get_array(df_train, 24)

        # full-scale clustering
        print('='*50)
        print('Fitting clusterer...')
        labels = self.model.fit_predict(arr_train_full)
        print('Fitting clusterer... Done!')
        print('='*50)
        print(collections.Counter(labels))
        print('='*50)
        print('')
        # plot_cluster(self.k, self.model, x_train_full, labels)

        label_df_train, label_df_test = get_label_df(df_train, df_test, labels, None)

        trans_mtx = get_transition_matrix(self.k, labels)
        # plot_trans_mtx(trans_mtx)

        sequence = make_sequence(self.k, self.model, 'dtw', df_test, label_df_train, label_df_test)

        df_clustered_list = seperate_dataset(self.k, label_df_train, df_train)

        return df_clustered_list, sequence, label_df_test



