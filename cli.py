import logging
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import help_functions as hp

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
#############################
# Data loading        #
#############################
logging.getLogger().setLevel(logging.INFO)
logging.info('Logging data')
# #Load data
df_actuals = pd.read_hdf('data_andres/stock_plan_actuals_for_assessing.h5')
df_pricing = pd.read_hdf('data_andres/stock_plan_4year_2month_pricing.h5')
#
# # Fill all the Null values from the actuals with -1 (TODO: Ask Andres why he filled them with -1)
df_actuals = df_actuals.fillna(-1)
# # Same for pricing
df_pricing = df_pricing.fillna(-1.0)
#
df_pricing_actuals = df_actuals.merge(df_pricing, how='left',
                                      on=['product_id', 'product_type_id',  'subsidiary_id', 'date'])
#
# # Keep only the NL/BE
SUBSIDIARIES = [1, 3]
df_pricing_actuals = df_pricing_actuals[df_pricing_actuals['subsidiary_id'].isin(SUBSIDIARIES)]
#df_pricing_actuals = pd.read_feather('df_pricing_actuals.feather')
# Start with one product id
#PRODUCT_ID = 508746
PRODUCT_ID = 888459
PRODUCT_ID = 508750
SUBSIDIARY = 1
df_pricing_actuals_pid_daily = df_pricing_actuals[(df_pricing_actuals['product_id'] == PRODUCT_ID)
                                            & (df_pricing_actuals['subsidiary_id'] == SUBSIDIARY)]
LAST_TRAIN_DATE = '2021-06-18'
# #output = hp.make_predictions_for(df_pricing_actuals_pid, LAST_TRAIN_DATE)
# df_pricing_actuals_pid_daily =  pd.read_feather('df_pricing_actuals_pid_daily_508746.feather')

#RUN_PIDS =  pd.read_csv('df_random_pids.csv').pids
logging.info('Finished loading and merging data data')

def run_for(last_train_date, subsidiary, product_ids, file_path, file_name, data=df_pricing_actuals, save=True):
    """
    Main function that calculates the forecast for a product id or more
    :param last_train_date: Last train date
    :param subsidiary: The subsidiary
    :param product_ids: A lis of product ids
    :param file_path: The path for svae the results
    :param file_name: The name of the file
    :param data: The whole dataset as given to Andres
    :param save: Whether to save the results or not
    :return:
    """
    start = time.time()
    # TODO: ENABLE THAT IF NEEDED
    #temp = data[data['date'] <= date]
    temp = data.copy()
    temp = temp[temp['subsidiary_id'] == subsidiary]
    LAST_TRAIN_DATE = last_train_date
    #all_pids = []
    # If empty run for all, else run for the specified products
    if len(product_ids) == 0:
        product_ids = temp.product_id.unique()
    else:
        product_ids = product_ids
    # This helps to save intermediate resulst
    batches = [product_ids[:]]
    for idx, batch in enumerate(batches):
        all_results = pd.DataFrame()
        results = {}
        # Loop over all product ids
        for product_id in tqdm(batch):
            df_pid_daily = temp[(temp['product_id'] == product_id)]
            if df_pid_daily.date.min()> pd.to_datetime(LAST_TRAIN_DATE):
                # Some product ids exists after the training date
                print('Skipping because date bigger than train date')
                y_ = 0
                X_ = pd.DataFrame()
                df_pricing_actuals_pid_weekly = pd.DataFrame()
            else:
                df_pricing_actuals_pid_weekly = hp.process_features(df_pid_daily, LAST_TRAIN_DATE, future_days=7)
            # all_pids.append(product_id)
            #print(df_pid_daily.product_id.unique())
                X_, y_ = hp.get_X_y(df_pricing_actuals_pid_weekly)
            cv_ = hp.estimate_model(X_, y_)
            all_data, predictions = hp.get_predictions(cv_, X_, df_pricing_actuals_pid_weekly, 7,
                                                       last_train_date=LAST_TRAIN_DATE)
            # results['product_id'] = product_id
            results[product_id] = {}
            if predictions.empty:
                # print(product_id)
                results[product_id]['prediction_date'] = pd.to_datetime(LAST_TRAIN_DATE) + timedelta(days=1)
                results[product_id]['predictions'] = -100
            else:
                results[product_id]['prediction_date'] = predictions.date[0]
                if predictions.predictions_7[0] < 1:
                    predictions = np.ceil(predictions.predictions_7[0])
                    results[product_id]['predictions'] = predictions
                else:
                    results[product_id]['predictions'] = predictions.predictions_7[0]
        if save:
            results = pd.DataFrame(results)
            all_results = pd.concat([all_results, results])
            FILE_PATH = file_path
            FILE_NAME = file_name + '_batch_{}.csv'.format(idx)
            # FILE_NAME = 'actual_predictions_NL_{}_correct_for_smaller_than_1_new_settings.csv'.format(LAST_TRAIN_DATE)
            all_results_for_csv = all_results.T
            all_results_for_csv['product_id'] = all_results_for_csv.index
            all_results_for_csv = all_results_for_csv.reset_index(drop=True)
            all_results_for_csv.to_csv(FILE_PATH + '/' + FILE_NAME, index=False)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    RUN_PIDS = pd.read_csv('df_random_pids.csv').pids
    LAST_TRAIN_DATE = '2021-06-18'
    # Path to write the results
    FILE_PATH = 'python_results'
    # Name of the file
    FILE_NAME = 'actual_predictions_NL_{}_glment_0_15000'.format(LAST_TRAIN_DATE)
    # If empty list than will run for all the product ids
    SUBISIDIARY = 1
    RUN_PIDS = []
    RUN_PIDS = [868632]
    run_for(LAST_TRAIN_DATE, SUBISIDIARY, RUN_PIDS,
            file_path=FILE_PATH, file_name=FILE_NAME)
