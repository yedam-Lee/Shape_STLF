import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.utils import get_array
from utils.constants import LOOK_BACK_CLASS as LB


def to_classification(data, sequence, look_back, df_weather):
    x, y = [], []

    in_start = 7

    for i in range(look_back, len(sequence)):
        two_day_before_total = data[in_start+(7-2)].sum()
        week_before_total = data[in_start].sum()
        week_before_cluster = sequence.iloc[in_start].label
        two_week_before_cluster = sequence.iloc[in_start-7].label
        day_of_week = sequence.iloc[i].name.weekday()
        month = sequence.iloc[i].name.month
        x.append([two_day_before_total, week_before_total, week_before_cluster, two_week_before_cluster, day_of_week, month])
        y.append(sequence.iloc[i].label)
        in_start += 1
    x, y = np.array(x), np.array(y)

    holidays = df_weather['holiday'].resample('D').mean()
    holidays = holidays.ravel()[look_back:]
    x_h = holidays.reshape(holidays.shape[0], 1)
    x = np.hstack([x, x_h])

    return x, y


class Classifier:

    def __init__(self):
        param_grid = {
            'n_estimators': [500, 1000],
            'criterion': ['gini', 'entropy'],
            'max_features': [5, 10, 15, 20, 25]
        }

        rf = RandomForestClassifier(random_state=42)
        self.model = GridSearchCV(estimator=rf, param_grid=param_grid,
                                  cv=3, n_jobs=5, verbose=0)

    def fit(self, train_test_clu, sequence, df_weather):
        df_train, df_test = train_test_clu[0], train_test_clu[1]

        train_hour, _ = get_array(df_train, 24)
        test_hour, _ = get_array(df_test, 24)
        train_test_hour = np.vstack([train_hour, test_hour])
        train_test_hour = train_test_hour.reshape(train_test_hour.shape[0], train_test_hour.shape[1], 1)

        x, y = to_classification(train_test_hour, sequence, LB, df_weather)

        y_enc = OneHotEncoder(categories='auto')
        y_enc.fit(y.reshape(-1, 1))
        y = y_enc.transform(y.reshape(-1, 1)).toarray()

        y_true = np.argmax(y, axis=1)
        for i in range(x.shape[1]):
            if i in [0, 1, 2, 3]:
                # day_before_total, week_before_total, week_before_cluster, two_week_before_cluster는 one-hot-vector 아님.
                x = np.concatenate((x, x[:, i].reshape(-1, 1)), axis=1)
            else:
                enc = OneHotEncoder(categories='auto')
                enc.fit(x[:, i].reshape(-1, 1))
                x_enc = enc.transform(x[:, i].reshape(-1, 1)).toarray()
                x = np.concatenate((x, x_enc), axis=1)
        x = x[:, 7:]

        scaler = StandardScaler()
        x = scaler.fit_transform(x.reshape(x.shape[0], x.shape[1]))
        x_train = x[:1216 - LB]
        y_train = y[:1216 - LB]
        x_test = x[1216 - LB:]
        y_test = y[1216 - LB:]
        y_true = y_true[1216 - LB:]

        print('='*50)
        print('Fitting classifier...')
        self.model.fit(x_train, y_train)
        print('Fitting classifier... Done!')
        print('='*50)
        # print('Best parameter is\n')
        # print(self.model.best_params_)
        # print('='*50)

        best_rf = self.model.best_estimator_
        # print('Scores \n')
        # print('Train:',best_rf.score(x_train, y_train))
        # print('Test:',best_rf.score(x_test, y_test))
        # print('='*50)

        y_pred = best_rf.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        print('Classifier Accuracy:', accuracy_score(y_true, y_pred))
        print('='*50)
        print('')

        test_clf = [x_test, y_test]

        return best_rf, test_clf
