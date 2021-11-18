import numpy as np
# Different penalties for Elastic Net based on the length of the training data
# Set of settings when the train data has more than 30 samples
ELASTIC_NET_SETTINGS_30 = {  # 'l1_ratio': [1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 2, 2],
    'fit_intercept': True,
    'alphas': [0.8],
    'random_state': 42,
    'cv': 3,
    'tol': 1e-8,
    'normalize': True,
    # 'n_jobs':4,
    'max_iter': 8000000

}
ELASTIC_NET_SETTINGS_OTHER = {  # 'l1_ratio': [1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 0, 0],
    'fit_intercept': True,
    'alpha': [0.8],
    'random_state': 42,
    'cv': 3,
    'tol': 1e-8,
    'normalize': True,
    'max_iter': 8000000
}
GLMNET_NET_SETTINGS_30 = {'penalties': np.array([1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 2, 2]).astype(float),
                          'lower_limit': np.array(
                              [-np.inf, -np.inf, 0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -np.inf,
                               -np.inf]).astype(float),
                          'upper_limit': np.array(
                              [0, np.inf, np.inf, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, np.inf, np.inf]).astype(float)
                          }
GLMNET_NET_SETTINGS_REST = {'penalties': np.array([1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 0, 0]).astype(float),
                            'lower_limit': np.array(
                                [-np.inf, -np.inf, 0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.3, -np.inf,
                                 -np.inf]).astype(float),
                            'upper_limit': np.array(
                                [0, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, np.inf, np.inf]).astype(float)
                            }
