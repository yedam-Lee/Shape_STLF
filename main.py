import sys
from utils.utils import read_dataset, evaluate
from utils.constants import NUMBER_OF_CLUSTERS as K


def fit_model():
    n_clusters = K
    print('')
    print('='*50)
    print("Let's start with {} clusters!".format(n_clusters))
    print('='*50)
    print('')

    train_test_clu, train_test_reg, df_weather = read_dataset()

    clusterer = create_clusterer(n_clusters)
    df_clustered_list, sequence, label_df_test = clusterer.fit(train_test_clu)

    classifier = create_classifier()
    best_classifier, test_clf = classifier.fit(train_test_clu, sequence, df_weather)

    regressor = create_regressor()
    best_model_set = regressor.fit(df_clustered_list, train_test_reg)

    pred_for_plot, true_for_plot = evaluate(label_df_test, train_test_reg, test_clf, best_classifier, best_model_set, sequence)

    return


def create_clusterer(n_clusters):
    from models import clusterer
    return clusterer.Clusterer(n_clusters)


def create_classifier():
    from models import classifier
    return classifier.Classifier()


def create_regressor():
    from models import regressor
    return regressor.Regressor()


if sys.argv[1] == 'run':

    fit_model()

    print('DONE')

# if __name__ == '__main__':
#
#     fit_model()
#
#     print('DONE')
