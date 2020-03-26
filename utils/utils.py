import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from workalendar.asia import SouthKorea
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def set_test_dates(test_df):
    test_dates = list(test_df.index.strftime('%Y-%m-%d'))
    test_dates = [datetime.strptime(x, '%Y-%m-%d') for x in test_dates]
    return test_dates


def plot_cluster(k, model, train_set, label):
    rows = (k//5)+1
    plt.figure(figsize=(15, rows*3))
    for i in range(k):
        plt.subplot(rows, 5, 1+i)
        for xx in train_set[label == i]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(model.cluster_centers_[i].ravel(), "C2--")
        plt.xlim(0, 23)
        plt.ylim(100, 1100)
        plt.xlabel('Time')
    plt.tight_layout()
    plt.show()


def plot_trans_mtx(trans_mat):
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(trans_mat, annot=True, cmap='YlOrRd')
    plt.title('Transitions', fontsize=14)
    plt.xlabel('To cluster')
    plt.ylabel('From cluster')
    plt.show()


def get_transition_matrix(k, label):
    mtx = np.zeros((k, k))
    for i in range(len(label)-1):
        for r in range(k):
            for c in range(k):
                if label[i] == r:
                    if label[i+1] == c:
                        mtx[r, c] += 1
                else:
                    break
    counts = mtx.sum(axis=1)
    for r in range(k):
        for c in range(k):
            mtx[r][c] /= counts[r]
            mtx[r][c] = round(mtx[r][c], 4)
    return mtx


def reshape_label_df(df, label):
    ldf = df.resample('D').mean().dropna()
    ldf['label'] = 0 if label is None else label
    ldf = pd.DataFrame(ldf.label)
    return ldf


def get_label_df(train_df, test_df, train_label, test_label):
    if train_df is None:
        train_ldf = None
    else:
        train_ldf = reshape_label_df(train_df, train_label)
    if test_df is None:
        test_ldf = None
    else:
        test_ldf = reshape_label_df(test_df, test_label)
    return train_ldf, test_ldf


def get_array(df, interval):
    array = df.values.reshape(int(len(df) / interval), interval)
    norm_arr = array.copy()
    norm_arr = norm_arr.reshape(norm_arr.shape[0], norm_arr.shape[1], 1)
    scalers = {}
    for i in range(array.shape[0]):
        scalers[i] = StandardScaler()
        norm_arr[i] = scalers[i].fit_transform(norm_arr[i])
    norm_arr = norm_arr.reshape(norm_arr.shape[0], norm_arr.shape[1])
    reshaped_arr = array.reshape(len(array), interval)
    return reshaped_arr, norm_arr


def get_holidays(cal, year):
    lst = []
    cal = cal.holidays(year)
    for tup in cal:
        lst.append(tup[0])
    return lst


def get_holidays_list():
    calendar_ko = SouthKorea()
    d = {}
    for y in range(2015, 2020):
        holidays = get_holidays(calendar_ko, y)
        d[y] = holidays
    # 임시공휴일
    temp_holidays = ['2015, 4, 28', '2015, 5, 1', '2015, 8, 14', '2015, 9, 29',
                     '2016, 2, 10', '2016, 4, 13', '2016, 5, 1', '2016, 5, 6',
                     '2017, 1, 30', '2017, 5, 1','2017, 5, 9', '2017, 10, 2', '2017, 10, 6',
                     '2018, 5, 1', '2018, 5, 7', '2018, 6, 13', '2018, 9, 26',
                     '2019, 5, 6']
    for h in temp_holidays:
        s = h.split(',')
        t = datetime.strptime(h, "%Y, %m, %d").date()
        d[int(s[0])].append(t)
    holiday_list = [v for lst in d.values() for v in lst]
    return holiday_list


def add_holiday(weather):
    is_holiday = []
    holiday_list = get_holidays_list()
    for i in range(len(weather)):
        boolean = weather.index[i] in holiday_list
        is_holiday.append(boolean)
    mask_holidays = (weather.index.weekday == 5) | (weather.index.weekday == 6) | is_holiday
    weather['holiday'] = mask_holidays
    weather['holiday'] = weather['holiday'].astype(int)
    return weather, holiday_list


def load_data():
    df = pd.read_csv("DW_wholedata.csv")
    df.date = df.date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S", errors='ignore')
    df.set_index('date', inplace=True)
    df_demand = df.resample('60T', label='left', closed='left').sum()
    df_demand = df_demand[df_demand.demand != 0]

    df_weather = pd.read_csv("DW_weather_hourly.csv")
    df_weather.date = df_weather.date.apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S", errors='ignore')
    df_weather.set_index('date', inplace=True)

    return df_demand, df_weather


def to_split(demand, weather):
    df_train_clu = demand.loc[:'2018-04-30']
    df_test_clu = demand.loc['2018-05-01':]

    df_concat = pd.concat([demand, weather], axis=1)
    df = df_concat[['temp', 'windspeed', 'humidity', 'holiday', 'demand']]
    df_train = df.loc[:'2018-04-30']
    df_test = df.loc['2018-05-01':]

    return df_train_clu, df_test_clu, df, df_train, df_test


def get_lagged_demand(df):
    lagged_list_h = [24,25,26,48,72,96,120,144,168]
    lagged_list_t = [24,48,72,96,120,144,168]
    lagged_list_r = [24,48,72]

    for c in df.columns:
        if c == 'holiday':
            continue
        elif c == 'demand':
            lagged_list = lagged_list_h
        elif c == 'temp':
            lagged_list = lagged_list_t
        else:
            lagged_list = lagged_list_r
        for sh in lagged_list:
            df['lagged_{}_{}'.format(c, sh)] = df[c].shift(sh)

    lagged_demand = df.dropna().drop(['temp', 'windspeed', 'humidity', 'demand'], axis=1)
    target_demand = df.dropna()[['demand']]['demand']

    return lagged_demand, target_demand


def to_regression(df, df_train, df_test):
    lagged_demand, target_demand = get_lagged_demand(df)

    x_train = lagged_demand[lagged_demand.index.isin(df_train.index)]
    x_test = lagged_demand[lagged_demand.index.isin(df_test.index)]

    y_train = target_demand[target_demand.index.isin(df_train.index)]
    y_test = target_demand[target_demand.index.isin(df_test.index)]

    feature_names = list(x_train.columns)

    return x_train, x_test, y_train, y_test, feature_names


def read_dataset():
    df_demand, df_weather = load_data()

    df_weather, holiday_list = add_holiday(df_weather)

    df_train_clu, df_test_clu, df, df_train, df_test = to_split(df_demand, df_weather)
    train_test_clu = [df_train_clu, df_test_clu]

    x_train_reg, x_test_reg, y_train_reg, y_test_reg, feature_names_reg = to_regression(df, df_train, df_test)
    train_test_reg = [x_train_reg, x_test_reg, y_train_reg, y_test_reg, feature_names_reg]

    return train_test_clu, train_test_reg, df_weather


# Performance Measures
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return mean_absolute_error(y_true, y_pred)


def evaluate(label_df_test, train_test_reg, test_clf, classifier, regressor_set, sequence):
    x_test_reg, y_test_reg = train_test_reg[1], train_test_reg[3]
    x_test_clf = test_clf[0]

    test_dates = set_test_dates(label_df_test)
    result_mape, result_rmse, result_mae = [], [], []

    pred_for_plot = list()
    true_for_plot = list()

    for i, dt in enumerate(test_dates):
        y_pred_clf = classifier.predict(x_test_clf[i].reshape(1, -1))
        y_pred_clf = np.argmax(y_pred_clf, axis=1)[0]
        selected_reg, sc = regressor_set[y_pred_clf]

        x_test = x_test_reg.loc[str(dt.date())]
        x_test_sc = sc.transform(x_test)

        test_ytrue = y_test_reg.loc[str(dt.date())]
        test_ypred = selected_reg.predict(x_test_sc)

        result_mape.append(mape(test_ytrue, test_ypred))
        result_rmse.append(rmse(test_ytrue, test_ypred))
        result_mae.append(mae(test_ytrue, test_ypred))

        pred_for_plot.append(test_ypred)
        true_for_plot.append(test_ytrue)

    print('=' * 50)
    print('Result')
    print('MAPE:', np.mean(result_mape))
    print('RMSE:', np.mean(result_rmse))
    print('MAE:', np.mean(result_mae))
    print('='*50)
    print('')

    return pred_for_plot, true_for_plot



