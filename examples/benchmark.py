import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from skl_sampling_bayesian_transformer import SamplingBayesianEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

# https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark
from skl_sampling_bayesian_transformer.sampling_bayesian_encoder import EncoderWrapper


def score_all_encoders_lr(train, target, encoders, feature_list, sampling_encoder):
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=97)
    for encoder in encoders:
        print_score_lr_encoder(encoder, feature_list, X_train, X_val, y_train, y_val)
    print_score_lr_sampling(sampling_encoder, X_train, X_val, y_train, y_val)


def print_score_lr_encoder(encoder, feature_list, X_train, X_val, y_train, y_val):
    print("Test {} : ".format(str(encoder).split('(')[0]), end=" ")
    train_enc = encoder.fit_transform(X_train[feature_list], y_train)
    val_enc = encoder.transform(X_val[feature_list])
    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr.fit(train_enc, y_train)
    lr_pred = lr.predict_proba(val_enc)
    score = auc(y_val, lr_pred)
    print("score: ", score)


def print_score_lr_sampling(encoder, X_train, X_val, y_train, y_val):
    print("Test {} : ".format(str(encoder).split('(')[0]), end=" ")
    lr = LogisticRegression(C=0.1, solver="lbfgs")  # , max_iter=1000)
    ew = EncoderWrapper(encoder, lr)
    ew.fit(X_train, y_train)
    lr_pred = ew.predict_proba(X_val)
    score = auc(y_val, lr_pred)
    print(" Sampling bayesian score: ", score)


def run_cat_in_the_cat_xp():
    train_raw = pd.read_csv('csv/input/cat-in-the-dat/train.csv')
    test_raw = pd.read_csv('csv/input/cat-in-the-dat/test.csv')
    target = train_raw['target']
    train_raw.drop(['target', 'id'], axis=1, inplace=True)
    test_raw.drop('id', axis=1, inplace=True)

    # remove feature_list?

    feature_list = list(train_raw.columns)
    encoder_list = [OrdinalEncoder(), WOEEncoder(), TargetEncoder(), MEstimateEncoder(), JamesSteinEncoder(),
                    LeaveOneOutEncoder(), CatBoostEncoder()]
    sampling_encoder = SamplingBayesianEncoder(cols=feature_list, n_draws=10)
    #score_all_encoders_lr(train_raw, target, encoder_list, feature_list, sampling_encoder)

    ## Cross val xp
    lr_params = {'solver': 'lbfgs', 'C': 0.1}
    results = list()
    train = train_raw[feature_list]
    test = test_raw[feature_list]

    for encoder in encoder_list:

        result = run_cv_lr(train, test, target,encoder, lr_params)
        results.append(result)

    sampling_encoder = SamplingBayesianEncoder(cols=feature_list, n_draws=10)
    result = run_cv_encoding_wraper(train, test, target, sampling_encoder, lr_params)
    results.append(result)

    results = pd.DataFrame(results)
    results['cv_mean'] = results['cv'].apply(lambda l: np.mean(l))
    results['cv_std'] = results['cv'].apply(lambda l: np.std(l))
    print(results[['label', 'cv_mean', 'cv_std']].head(9))


def run_cv_encoding_wraper(train, test, target, sampling_encoder, lr_params):
    label = str(sampling_encoder).split('(')[0]
    kf = KFold(n_splits=5)
    fold_splits = kf.split(train, target)
    lr = LogisticRegression(**lr_params)
    wrapper = EncoderWrapper(sampling_encoder, lr)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started {} fold {}/5'.format(label, i))
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        wrapper.fit(dev_X, dev_y)
        pred_val_y = wrapper.predict_proba(val_X)
        pred_test_y = wrapper.predict_proba(test)

        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_score = auc(val_y, pred_val_y)
        cv_scores.append(cv_score)
        print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label, 'train': pred_train, 'test': pred_full_test, 'cv': cv_scores}
    return results


def run_cv_lr(train, test, target, encoder, lr_params):
    label = str(encoder).split('(')[0]
    kf = KFold(n_splits=5)
    fold_splits = kf.split(train, target)
    model = LogisticRegression(**lr_params)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started {} fold {}/5'.format(label, i))
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        dev_X_enc = encoder.fit_transform(dev_X, dev_y)
        val_X_enc = encoder.transform(val_X)
        model.fit(dev_X_enc, dev_y)
        pred_val_y = model.predict_proba(val_X_enc)[:, 1]
        test_enc = encoder.transform(test)
        pred_test_y = model.predict_proba(test_enc)[:, 1]

        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_score = auc(val_y, pred_val_y)
        cv_scores.append(cv_score)
        print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label, 'train': pred_train, 'test': pred_full_test, 'cv': cv_scores}
    return results

if __name__ == '__main__':
    run_cat_in_the_cat_xp()
