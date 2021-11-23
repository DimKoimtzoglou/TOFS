import logging
from datetime import timedelta
import logging
import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
import gc
num_cores = multiprocessing.cpu_count()
np.random.seed(123)
NUMBER_LAGS = 7
import settings as s
# # If you want to
# client = Client(processes=False)             # create local cluster
import glmnet_python
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def estimate_model(X, y):
    """
    Function that estimates a model using cross validation
    :param X: Train data
    :param y: Target
    :return: An optimized cv_ model
    """
    if X.shape[0] == 0:
        cv_ = None
    else:
        if X.shape[0] > 30:
            penalties_ = s.GLMNET_NET_SETTINGS_30['penalties']
            lower_limit = s.GLMNET_NET_SETTINGS_30['lower_limit']
            upper_limit = s.GLMNET_NET_SETTINGS_30['upper_limit']
            limits = np.array([lower_limit, upper_limit])
        else:
            penalties_ = s.GLMNET_NET_SETTINGS_REST['penalties']
            lower_limit = s.GLMNET_NET_SETTINGS_REST['lower_limit']
            upper_limit = s.GLMNET_NET_SETTINGS_REST['upper_limit']
            limits = np.array([lower_limit, upper_limit])

        # Last observations have higher weight
        weights = [1.0] * (X.shape[0] - 3) + [2.0] * 3
        # Elastic Net model
        # elasticNet_model = clone(ElasticNetCV(**s.ELASTIC_NET_SETTINGS_30))
        # elasticNet_model = clone(ElasticNetCV(cv=3 ,random_state=42))
        # TODO: Current version of Elastic Net does not support weights --> Need to upgrade
        # elasticNet_cv = elasticNet_model.fit(X, y, sample_weight=weights)
        try:
            # with joblib.parallel_backend('dask'):
            # cv_ = elasticNet_model.fit(X, y)
            ORDER_COLUMNS = ['weekly_price_avg', 'prod_age', 'weekly_lowest_tier_one_avg', 'lag_1', 'lag_2',
                            'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'step_last_21',
                            'step_last_15']
            X = X[ORDER_COLUMNS]
            np.random.seed(123)
            cv_ = cvglmnet(x=X.to_numpy().copy(),
                           y=y.values.astype(float),
                           penalty_factor=penalties_,
                          # lambdau  = np.array([539, 540.258]),
                           alpha=0.8,
                           intr=True,
                           weights=np.array(weights).astype(float),
                           cl=limits,
                           nfolds=3,
                           dfmax=np.array([5]),
                           maxit=8000000,
                           thresh=1e-8
                           )
        except ValueError as e:
            logging.warning('Error in CV')
            cv_= None
    return cv_


def get_predictions(cv_model, train_data, all_data, predictions_ahead, last_train_date):
    """
    Main function to return the predictions
    :param cv_model: The trained model
    :param train_data: train data
    :param all_data: All the data used for to generate the train data
    :param predictions_ahead: Forecast horizon
    :param last_train_date: The last dat of training
    :return:
    Notes: In this approach, we correct the predictions in 3 different stages
    Correction 1: Correction using the residuals of the last 7 days
                Reasoning behind: Correct for bias in the residuals
    Correction 2: Correction using the residuals of the last 21 days
                Reasoning behind: Re-Correct the predictions
    Correction 3: Corrections using the mean of the last 7 days actuals
             Reasoning behind: Provides stability to the forecast and we avoid sudden changes
    """
    if cv_model:
        # fitting = cv_model.predict(train_data)
        ORDER_COLUMNS = ['weekly_price_avg', 'prod_age', 'weekly_lowest_tier_one_avg', 'lag_1', 'lag_2',
                         'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'step_last_21',
                         'step_last_15']
        train_data = train_data[ORDER_COLUMNS]
        #all_data = all_data[ORDER_COLUMNS]
        fitting = cvglmnetPredict(cv_model, newx=train_data.to_numpy().copy())
#        fitting = glmnetPredict(cv_model, newx=train_data.to_numpy().copy())
        all_data['fitted'] = fitting
        all_data = all_data.reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)
        # No clue why Andres does this need to ask
        if all_data.weekly_actual_sum.sum() > 0:
            find_lastSale = all_data[all_data.weekly_actual_sum > 0].index[-1]
        else:
            find_lastSale = 1
        iniLastObs = np.max([np.min([find_lastSale, all_data.shape[0] - predictions_ahead]), 1])
        all_data['x_hat'] = all_data['weekly_actual_sum'] - all_data['fitted']
        x_hat = all_data['x_hat'].iloc[iniLastObs - 1:]
        # Sum of residuals
        sum_x_hat = np.sum(x_hat)
        mean_last_observations = np.mean(all_data['weekly_actual_sum'].iloc[iniLastObs - 1:])
        # Correction 1
        # If you are residuals are <0 => you are over forecasting
        # If the mean of the last observations < 50 => Small time series
        # The parameters multiplier and stdcomponent will be used to create normal distribution. The idea behind it is that
        # if we have signs of over estimation in the resiudals than we should correct it
        # The w1, w2 parameters are used to weight the predictions from the glmnet and the lag used
        if (sum_x_hat < 0) & (mean_last_observations < 50):
            # TODO: Fix this/Also exists in Andres approach
            if sum_x_hat == 0:
                multiplier = 4
                stdcomponent = np.std(x_hat) / 4
                w1 = 0.9
                w2 = 0.1
            else:
                multiplier = 3
                stdcomponent = np.std(x_hat) / 3
                w1 = 0.8
                w2 = 0.2
        else:
            # This is the case where the residuals do not show a a sign of bias so no need to correct for it
            multiplier = 0
            stdcomponent = 0
            w1 = 0.5
            w2 = 0.5
        # Add the random component/Volatility on sales
        mu, sigma = multiplier * np.mean(x_hat), stdcomponent
        v = np.random.normal(mu, sigma, predictions_ahead)
        # Get the last training values
        last_train_values = train_data.tail(1)
        last_train_values['prod_age'] = np.log(np.exp(last_train_values['prod_age']) + 1)
        start_predictions = pd.to_datetime(last_train_date) + timedelta(days=1)
        prediction_dates = pd.date_range(start_predictions, periods=predictions_ahead)
        predictions = []
        # TODO: Ugly change it as soon as possible
        # Prepare the prediction data/Initialize with
        # Make sure that the order of the features is same as the train data
        weekly_price_avg = last_train_values['weekly_price_avg']
        weekly_lowest_tier_one_avg = last_train_values['weekly_lowest_tier_one_avg']
        lag_1 = all_data[all_data['date'] <= last_train_date]['weekly_actual_sum']
        lag_1 = lag_1.tail(1)
        lag_2 = last_train_values['lag_1']
        lag_3 = last_train_values['lag_2']
        lag_4 = last_train_values['lag_3']
        lag_5 = last_train_values['lag_4']
        lag_6 = last_train_values['lag_5']
        lag_7 = last_train_values['lag_6']
        step_last_21 = last_train_values['step_last_21']
        step_last_15 = last_train_values['step_last_15']
        prod_age = last_train_values['prod_age']
        df_prediction = pd.DataFrame({'weekly_price_avg': weekly_price_avg,
                                      'prod_age': prod_age,
                                      'weekly_lowest_tier_one_avg': weekly_lowest_tier_one_avg,
                                      'lag_1': lag_1,
                                      'lag_2': lag_2,
                                      'lag_3': lag_3,
                                      'lag_4': lag_4,
                                      'lag_5': lag_5,
                                      'lag_6': lag_6,
                                      'lag_7': lag_7,
                                      'step_last_21': step_last_21,
                                      'step_last_15': step_last_15,
                                      })
        for idx, day in enumerate(prediction_dates):
            # elasticnet_prediction = cv_model.predict(df_prediction)
            elasticnet_prediction = cvglmnetPredict(cv_model, newx=df_prediction.to_numpy().copy())
            #elasticnet_prediction = glmnetPredict(cv_model, newx=df_prediction.to_numpy().copy())

            #print(elasticnet_prediction)
            prediction = w2 * np.max([elasticnet_prediction[0] + v[idx], 0.0]) + df_prediction['lag_1'] * w1
            predictions.append(prediction.values[0])
            # Update the df prediction dataframe
            df_prediction_cp = df_prediction.copy()
            df_prediction['prod_age'] = np.log(np.exp(last_train_values['prod_age']) + 1)
            df_prediction['lag_1'] = np.ceil(prediction.values[0])
            df_prediction['lag_2'] = df_prediction_cp['lag_1']
            df_prediction['lag_3'] = df_prediction_cp['lag_2']
            df_prediction['lag_4'] = df_prediction_cp['lag_3']
            df_prediction['lag_5'] = df_prediction_cp['lag_4']
            df_prediction['lag_6'] = df_prediction_cp['lag_5']
            df_prediction['lag_7'] = df_prediction_cp['lag_6']
        # Correction 2
        iniLastObs = np.max([np.min([find_lastSale, all_data.shape[0] - 21]), 1])
        x_hat = all_data['x_hat'].iloc[iniLastObs - 1:]
        mu, sigma = np.mean(x_hat), np.std(x_hat)
        v = np.random.normal(mu, sigma, predictions_ahead)
        # For the key 824863 NL if you change v with those values then the result is the same WITH JULIA
        # v= np.array([-46.02754763498439,
        #     -14.470340754022867,
        #     -312.0933055056326,
        #     -175.6291803742907,
        #     -210.65555921132753,
        #     -35.79823464572249,
        #      53.986721301803215])
        # For the key 865474 BE if you change v with those values then the result is the same WITH JULIA
        # v = np.array([  0.8156827734679348,
        #      -15.738750989554019,
        #      -16.167517101298586,
        #      -15.803117749837238,
        #      -14.067289935181188,
        #      -10.772371540727697,
        #      -13.978731197648052])
        # 694030/NL
        # v = np.array(  [15.728082212079617,
        #                  -5.562543056146005,
        #                 -74.21757274192805,
        #                 10.427201677739003,
        #                 -26.627725987479153,
        #                 -36.79987793480082,
        #                 34.628866083309546])
        # v_seed = []
        # for seed_ in range(1,100):
        #     np.random.seed(seed_)
        #     v_per = np.random.normal(mu, sigma, predictions_ahead)
        #     v_seed.append(v_per)
        # v = np.mean(v_seed, axis=0)
        mean_of_last_7days = np.mean(all_data['weekly_actual_sum'].iloc[-(predictions_ahead + 1):])
        v2 = [mean_of_last_7days] * predictions_ahead
        v_predictions = predictions + v
        # correct for zeros
        v_predictions[v_predictions < 0] = 0
        v_predictions = (v_predictions + v2) / 2
        v_predictions = np.round(v_predictions, 0)
        last_21d = all_data.iloc[iniLastObs - 1:]
        zero_values_last21 = last_21d[last_21d['weekly_actual_sum'] == 0]
        non_zero_values_last21 = last_21d[last_21d['weekly_actual_sum'] > 0]
        # This is used to correct for low sellers
        if (zero_values_last21.shape[0] / last_21d.shape[0]) > 0.9:
            n, p = 1, non_zero_values_last21.shape[0] / np.min([train_data.shape[0] - iniLastObs, 7 * 12])
            v2 = np.random.binomial(n, p, predictions_ahead)
            idx_binom = np.where(v2 > 0)
            if len(idx_binom) > 0:
                for idx in list(idx_binom[0]):
                    try:
                        if v_predictions[idx] == 0:
                            v_predictions[idx] = 1
                    except Exception as e:
                        print('Raised exception')
                        print(v_predictions)
        all_predictions = pd.DataFrame({'predictions_7': [sum(v_predictions / 7)],
                                        'date': [prediction_dates[0]]})
    else:
        all_predictions = pd.DataFrame()
        all_data = pd.DataFrame()
    return all_data, all_predictions


def get_X_y(df_weekly_data):
    """
    Split the data in X,y
    :param df_weekly_data:
    :return: X,y
    """
    df_weekly_data_cp = df_weekly_data.copy()
    if not df_weekly_data_cp.empty:
        y_ = df_weekly_data_cp.weekly_actual_sum
        X_ = df_weekly_data_cp.copy()
        X_.drop(['weekly_actual_sum', 'date'], inplace=True, axis=1)
    else:
        logging.info('Not enough data')
        y_ = 0
        X_ = pd.DataFrame()
    return X_, y_

def process_features(daily_data, last_train_date, future_days=7):
    """
    Function that creates the fature set
    :param daily_data_cp: The daily data
    :param last_train_date: Last train days
    :param future_days: The forecast horizon
    :return: The feature set
    """
    # TODO: Remove this part it's only useful where you load the data from the feeather files
    if 'index' in daily_data.columns:
        daily_data = daily_data.drop('index', axis=1)
    daily_data_cp = daily_data.copy()
    daily_data_cp = daily_data_cp.sort_values(by='date')
    daily_data_cp = daily_data_cp.reset_index(drop=True)
    # I do not get why he is doing this but let's see
    last_train_date_plus_7 = pd.to_datetime(last_train_date) + timedelta(days=7)
    try:
        idx_last_date_plus_7 = daily_data_cp.index[daily_data_cp['date'] == last_train_date_plus_7].tolist()[0] + 1
    except Exception as e:
        idx_last_date_plus_7 = daily_data_cp.index[daily_data_cp['date'] <= last_train_date_plus_7].tolist()[-1] + 1
    daily_data_cp = daily_data_cp.iloc[0:idx_last_date_plus_7]
    # Product ids with one single line of data
    if daily_data.shape[0]==1:
        daily_data_cp = daily_data_cp.iloc[:]
    else:
        daily_data_cp = daily_data_cp.iloc[:-7]
    try:
        assert (daily_data_cp.date.max() <= pd.to_datetime(last_train_date))
    except AssertionError:
        logging.error('Assertion failed')
    #daily_data = daily_data[daily_data['date'] <= last_train_date]
    # Correct for negative values
    daily_data_cp['lowest_tier_one'] = np.where(daily_data_cp['lowest_tier_one'] < 0, 0, daily_data_cp['lowest_tier_one'])
    daily_data_cp['lowest_tier_one'] = daily_data_cp['lowest_tier_one'].fillna(0)
    daily_data_cp['price'] = daily_data_cp['price'].fillna(0)
    daily_data_cp = daily_data_cp.sort_values(by='date')
    if sum(daily_data_cp.actual_intermediate) > 0:
        # Find the first non-zero index in the actuals TODO:Can you find a better way here?
        actual_intermediate_array = np.array(daily_data_cp.actual_intermediate)
        daily_data_cp['actual'] = daily_data_cp.actual_intermediate
        non_zero_idx = np.nonzero(actual_intermediate_array)[0][0] - 8
        initial_val = np.max([non_zero_idx, 8])
        # Create weekly target
        daily_data_cp['weekly_actual_sum'] = daily_data_cp['actual'].rolling(window=7).sum().fillna(0)
        daily_data_cp['weekly_actual_sum'] = pd.to_numeric(daily_data_cp['weekly_actual_sum'], downcast="integer")
        # Create weekly pricing features
        daily_data_cp['weekly_price_avg'] = daily_data_cp['price'].rolling(window=7).mean()
        daily_data_cp['weekly_lowest_tier_one_avg'] = daily_data_cp['lowest_tier_one'].rolling(window=7).mean()
        for lag in range(1, NUMBER_LAGS + 1):
            feature = 'lag_{}'.format(lag)
            daily_data_cp[feature] = daily_data_cp['weekly_actual_sum'].shift(lag)
            # Fill the NaN with zero
            daily_data_cp[feature].fillna(0, inplace=True)
        daily_data_cp['step_last_21'] = 0
        daily_data_cp['step_last_15'] = 0
        if daily_data_cp.shape[0] >= 21:
            daily_data_cp.step_last_21.iloc[-21:] = 1
            daily_data_cp.step_last_15.iloc[-15:] = 1
        elif daily_data_cp.shape[0] >= 15:
            daily_data_cp.step_last_21.iloc[-1:] = 1
            daily_data_cp.step_last_15.iloc[-15:] = 1
        else:
            daily_data_cp.step_last_21.iloc[-1:] = 1
            daily_data_cp.step_last_15.iloc[-2:] = 1
        # Get the data after the first non-zero sale
        daily_data_cp = daily_data_cp.iloc[initial_val - 1:, :]
        daily_data_cp['prod_age'] = np.log(range(1, daily_data_cp.shape[0] + 1))
        keep_columns = ['date', 'weekly_actual_sum', 'weekly_price_avg', 'weekly_lowest_tier_one_avg',
                        'lag_1', 'lag_2', 'lag_3', 'lag_4',
                        'lag_5', 'lag_6', 'lag_7', 'step_last_21', 'step_last_15', 'prod_age']
        daily_data_cp = daily_data_cp[keep_columns]
        return daily_data_cp
    else:
        logging.warning('Feature engineering will return an empty dataframe')
        return pd.DataFrame()

def predictions_per_key(product_id, df_subsidiary, last_train_date):
    """
    Function that calculates the predictions per key
    :param product_id:
    :param df_subsidiary:
    :param last_train_date:
    :return:
    """

    daily_data = df_subsidiary[(df_subsidiary['product_id'] == product_id)]
    daily_data = daily_data.sort_values('date')
    results = {}
    if daily_data.date.min() > pd.to_datetime(last_train_date) or daily_data.shape[0]==0:
        # Some product ids exists after the training date
        print('Skipping because date bigger than train date or empty dataframe')
        y_ = 0
        X_ = pd.DataFrame()
        df_pricing_actuals_pid_weekly = pd.DataFrame()
    else:
        df_pricing_actuals_pid_weekly = process_features( daily_data, last_train_date, future_days=7)
        # all_pids.append(product_id)
        # print(df_pid_daily.product_id.unique())
        X_, y_ = get_X_y(df_pricing_actuals_pid_weekly)
    cv_ = estimate_model(X_, y_)
    all_data, predictions = get_predictions(cv_, X_, df_pricing_actuals_pid_weekly, 7,
                                               last_train_date=last_train_date)
    # results['product_id'] = product_id
    results[product_id] = {}
    if predictions.empty:
        # print(product_id)
        results[product_id]['prediction_date'] = pd.to_datetime(last_train_date) + timedelta(days=1)
        results[product_id]['predictions'] = -100
    else:
        results[product_id]['prediction_date'] = predictions.date[0]
        if predictions.predictions_7[0] < 1:
            predictions = np.ceil(predictions.predictions_7[0])
            results[product_id]['predictions'] = predictions
        else:
            results[product_id]['predictions'] = predictions.predictions_7[0]
    del cv_, all_data, predictions, X_, df_pricing_actuals_pid_weekly
    gc.collect()
    return pd.DataFrame(results).T
