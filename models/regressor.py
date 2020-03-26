import numpy as np

from xgboost import XGBRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

class Regressor:

    def __init__(self):
        self.model_set = dict()

        alpha_set = 10.0 ** -np.arange(0, 6)
        parameter_mlp = {
            'hidden_layer_sizes': [(10), (10, 10), (10, 10, 10), (30), (30, 30), (30, 30, 30)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': alpha_set,
            'learning_rate': ['constant', 'adaptive']
        }
        mlp = MLPRegressor(max_iter=10000)
        grid_mlp = GridSearchCV(mlp, parameter_mlp, n_jobs=5, cv=5, scoring='neg_root_mean_squared_error')
        self.model_set['mlp'] = grid_mlp

        parameter_svr = {'kernel': ['linear', 'poly', 'rbf'],
                         'C': [0.01, 1, 100],
                         'gamma': ['auto', 'scale']
                         }
        svr = SVR()
        grid_svr = GridSearchCV(svr, parameter_svr, n_jobs=5, cv=5, scoring='neg_root_mean_squared_error')
        self.model_set['svr'] = grid_svr

        xgb = XGBRegressor(learning_rate=0.1,
                           max_depth=5,
                           n_estimators=500,
                           silent=True)
        self.model_set['xgb'] = xgb

    def fit(self, df_clustered_list, train_test_reg):

        best_model_set = dict()

        x_train, y_train = train_test_reg[0], train_test_reg[2]

        for i in range(len(df_clustered_list)):
            import copy
            model_set = copy.deepcopy(self.model_set)
            mlp = model_set['mlp']
            svr = model_set['svr']
            xgb = model_set['xgb']

            print('=' * 50)
            print('For cluster', i)
            print('')

            df_clustered = df_clustered_list[i][0]
            x_train_clu = x_train[x_train.index.isin(df_clustered.index)]
            y_train_clu = y_train[y_train.index.isin(df_clustered.index)]

            sc = StandardScaler()
            x_train_clu_sc = sc.fit_transform(x_train_clu)
            y_train_value = y_train_clu.values.ravel()

            print('Fitting MLR...')
            mlp.fit(x_train_clu_sc, y_train_value)
            print('Fitting MLR... Done!')
            print('')
            print('Fitting SVM...')
            svr.fit(x_train_clu_sc, y_train_value)
            print('Fitting SVM... Done!')
            print('')
            print('Fitting XGB...')
            xgb.fit(x_train_clu_sc, y_train_value)
            xgb_score = cross_val_score(xgb, x_train_clu_sc, y_train_value, cv=5, scoring='neg_root_mean_squared_error')
            print('Fitting XGB... Done!')
            print('')

            scores = [mlp.best_score_, svr.best_score_, xgb_score.mean()]

            best_model_idx = np.argmax(scores)
            if best_model_idx == 0:
                best_model = mlp
            elif best_model_idx == 1:
                best_model = svr
            else:
                best_model = xgb
            best_model_set[i] = (best_model, sc)

        return best_model_set



