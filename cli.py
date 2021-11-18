import logging
import sys
import time
import warnings
from datetime import timedelta
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()
import numpy as np
import pandas as pd
from tqdm import tqdm

import help_functions as hp

if not sys.warnoptions:
    warnings.simplefilter("ignore")

SUBSIDIARY_DICT = {'1':'NL', '3':'BE'}
#############################
# Data loading        #
#############################
logging.getLogger().setLevel(logging.INFO)
logging.info('Logging data')
# # #Load data
# df_actuals = pd.read_hdf('data_andres/stock_plan_actuals_for_assessing.h5')
# df_pricing = pd.read_hdf('data_andres/stock_plan_4year_2month_pricing.h5')
# #
# # # Fill all the Null values from the actuals with -1 (TODO: Ask Andres why he filled them with -1)
# df_actuals = df_actuals.fillna(-1)
# # # Same for pricing
# df_pricing = df_pricing.fillna(-1.0)
# #
# df_pricing_actuals = df_actuals.merge(df_pricing, how='left',
#                                       on=['product_id', 'product_type_id',  'subsidiary_id', 'date'])
# #
# # # Keep only the NL/BE
# SUBSIDIARIES = [1, 3]
# df_pricing_actuals = df_pricing_actuals[df_pricing_actuals['subsidiary_id'].isin(SUBSIDIARIES)]
df_pricing_actuals = pd.read_feather('df_pricing_actuals.feather')
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

def run_for(last_train_date, subsidiary, product_ids, file_path, file_name,
            data=df_pricing_actuals, parallelize = True, save=True):
    """
    Main function that calculates the forecast for a product id or more
    :param last_train_date: Last train date
    :param subsidiary: The subsidiary
    :param product_ids: A lis of product ids
    :param file_path: The path for svae the results
    :param file_name: The name of the file
    :param data: The whole dataset as given to Andres
    :param parallelize: To parallelize or not
    :param save: Whether to save the results or not
    :return:
    """
    start = time.time()
    # TODO: ENABLE THAT IF NEEDED
    #temp = data[data['date'] <= date]
    temp_subsidiary = data.copy()
    temp_subsidiary = temp_subsidiary[temp_subsidiary['subsidiary_id'] == subsidiary]
    LAST_TRAIN_DATE = last_train_date
    #all_pids = []
    # If empty run for all, else run for the specified products
    if len(product_ids) == 0:
        logging.info('Running for all')
        product_ids = temp_subsidiary.product_id.unique()
    else:
        product_ids = product_ids
    if parallelize:
        logging.info('Parallelize the process')
        predictions_dict = Parallel(n_jobs=-1, backend='loky')(
            delayed(hp.predictions_per_key)(product_id, temp_subsidiary[(temp_subsidiary['product_id'] == product_id)], LAST_TRAIN_DATE)
            for product_id in tqdm(product_ids[:]))
    else:
        for product_id in tqdm(product_ids[:]):
            predictions_dict = hp.predictions_per_key(product_id, temp_subsidiary[(temp_subsidiary['product_id'] == product_id)], LAST_TRAIN_DATE)
            print(predictions_dict)
    if save:
        all_results = pd.concat(predictions_dict)
        FILE_PATH = file_path
        FILE_NAME = file_name + '_{}.csv'.format(SUBSIDIARY_DICT[str(subsidiary)])
        all_results_for_csv = all_results
        all_results_for_csv['product_id'] = all_results_for_csv.index
        all_results_for_csv = all_results_for_csv.reset_index(drop=True)
        all_results_for_csv.to_csv(FILE_PATH + '/' + FILE_NAME, index=False)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    # Few product ids to test for
    RUN_PIDS = pd.read_csv('df_random_pids.csv').pids
    NUMBER_OF_DAYS = 1
    START_DATE = '2021-06-26'
    date_range = pd.date_range(START_DATE, periods=NUMBER_OF_DAYS)
    for date_ in date_range:
        for subsidiary in [1, 3]:
            RUN_PIDS = [865474]
            RUN_PIDS = [694030]
            RUN_PIDS = [813637]
            RUN_PIDS= [871242]
            RUN_PIDS = [868402, 824863, 869587, 873097]
            SAVE_RESULTS = False
            PARALLELIZE = True
            # If empty list than will run for all the product ids
            #RUN_PIDS = []
            LAST_TRAIN_DATE = date_.strftime("%Y-%m-%d")
            FILE_PATH = 'python_results/python_results_set_seed'
            FILE_NAME = 'actual_predictions_{}_glmnet_no_seed_alpha_08_rs42'.format(LAST_TRAIN_DATE)
            print('Running for {} and subsidiary {}'.format(date_, subsidiary))
            run_for(LAST_TRAIN_DATE, subsidiary, RUN_PIDS,
                    file_path=FILE_PATH,
                    file_name=FILE_NAME,
                    parallelize=PARALLELIZE,
                    save=SAVE_RESULTS)
