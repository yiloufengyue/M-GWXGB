"""
Code for M-GWXGB model submitted to Annals of AAG
Author: Fan Gao, Sylvia He, and Mei-po Kwan
Date: June 01, 2025
"""

from itertools import product
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
import re
import math

class XGBoostTrainer_1:
    def __init__(self, n_splits=5, random_state=42, n_jobs=-1):
        self.best_model = None
        self.best_params = None
        self.oof_predictions = None
        self.cv_scores = []
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_iterations_per_fold = []
        self.oof_shap_values = None
        self.n_jobs = n_jobs

    def _train_fold(self, params, X, y, train_idx, val_idx, fold_num):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

        watchlist = [(dvalid_fold, 'eval')]

        model = xgb.train(params,
                         dtrain_fold,
                         num_boost_round=1000,
                         evals=watchlist,
                         early_stopping_rounds=30,
                         verbose_eval=False)

        return {
            'fold_num': fold_num,
            'score': model.best_score,
            'model': model,
            'val_idx': val_idx
        }

    def objective(self, params, X, y):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])

        param_use = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            **params
        }

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X, y))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_fold)(
                param_use, X, y, train_idx, val_idx, fold_num
            )
            for fold_num, (train_idx, val_idx) in enumerate(folds)
        )

        cv_scores = [res['score'] for res in results]
        avg_cv_score = np.mean(cv_scores)
        
        return {'loss': avg_cv_score, 'status': STATUS_OK}

    def _cv_fold(self, X, y, train_idx, val_idx, fold_num):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

        watchlist = [(dtrain_fold, 'train'), (dvalid_fold, 'eval')]

        fold_model = xgb.train(self.best_params,
                             dtrain_fold,
                             num_boost_round=2000,
                             evals=watchlist,
                             early_stopping_rounds=50,
                             verbose_eval=100 if fold_num == 0 else False)

        best_iteration = fold_model.best_iteration
        fold_score = fold_model.best_score

        oof_preds_fold = fold_model.predict(dvalid_fold, iteration_range=(0, best_iteration))
        
        fold_explainer = shap.TreeExplainer(fold_model)
        fold_shap_value = fold_explainer.shap_values(dvalid_fold)

        return {
            'fold_num': fold_num,
            'best_iteration': best_iteration,
            'score': fold_score,
            'val_idx': val_idx,
            'oof_pred': oof_preds_fold,
            'shap_values': fold_shap_value,
            'model': fold_model
        }

    def calcu_oof_and_shap(self, X, y, tune=True, max_evals=50):
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if tune or self.best_params is None:
            self.tune_params(X, y, max_evals=max_evals)
        elif not tune and self.best_params is None:
            raise ValueError("Cannot train without tuning if best_params are not already set.")

        print(f"\nStarting {self.n_splits}-Fold CV for OOF predictions (parallel)...")
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X, y))

        self.oof_predictions = np.zeros(X.shape[0])
        self.oof_shap_values = np.zeros(X.shape)
        self.cv_scores = []
        self.best_iterations_per_fold = []

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._cv_fold)(X, y, train_idx, val_idx, fold_num)
            for fold_num, (train_idx, val_idx) in enumerate(folds)
        )

        for res in results:
            self.best_iterations_per_fold.append(res['best_iteration'])
            self.cv_scores.append(res['score'])
            self.oof_predictions[res['val_idx']] = res['oof_pred']
            self.oof_shap_values[res['val_idx']] = res['shap_values']

        oof_rmse = np.sqrt(mean_squared_error(y, self.oof_predictions))
        oof_r2 = r2_score(y, self.oof_predictions)
        
        print("\n--- Training Final Model on Full Data ---")
        final_num_boost_round = int(np.median(self.best_iterations_per_fold))
        dfull = xgb.DMatrix(X, label=y)

        self.best_model = xgb.train(self.best_params,
                                  dfull,
                                  num_boost_round=final_num_boost_round,
                                  verbose_eval=100)

        print("Final model training complete.")
        return self.best_model, self.oof_predictions, oof_rmse, oof_r2, self.oof_shap_values

    def tune_params(self, X, y, max_evals=50):
        search_space = {
            "max_depth": hp.quniform('max_depth', 3, 8, 1),
            "learning_rate": hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            "subsample": hp.uniform('subsample', 0.6, 1.0),
            "colsample_bytree": hp.uniform('colsample_bytree', 0.6, 1.0),
            "min_child_weight": hp.quniform('min_child_weight', 1, 6, 1),
            "gamma": hp.uniform('gamma', 0, 0.5),
            "reg_alpha": hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),
            "reg_lambda": hp.loguniform('reg_lambda', np.log(0.1), np.log(10.0)),
        }

        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, X, y),
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state)
        )

        self.best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': int(best['min_child_weight']),
            'gamma': best['gamma'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.random_state
        }
        print("\nBest parameters found:")
        print(self.best_params)
        return self.best_params

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.best_model.predict(dmatrix)

    def compute_shap_values(self, X):
        explainer = shap.TreeExplainer(self.best_model)
        dmatrix = xgb.DMatrix(X)
        return explainer.shap_values(dmatrix)

class GeoWeightedXGBoostTrainer:
    def __init__(self, data, target, locations, k_nearest=100, n_jobs=-1, n_clusters=100, use_full_sample=True):
        self.data = data
        self.target = target
        self.locations = np.array(locations)
        self.k_nearest = k_nearest
        self.n_jobs = n_jobs
        self.n_clusters = n_clusters
        self.models = []
        self.cluster_centers = None
        self.test_indices_list = []
        self.test_predictions_list = []
        self.saved_model = None
        self.use_full_sample = use_full_sample
        self.cluster_labels = None

    def search_optimal_k_nearest(self, k_range=(125, 2400, 300), max_trials=10):
        self.cluster_centers = self._cluster_data()
        min_error = float('inf')
        optimal_k = self.k_nearest
        consecutive_no_improve = 0

        for k in range(k_range[0], k_range[1] + 1, k_range[2]):
            self.k_nearest = k
            self.fit(use_full_sample=False)
            test_error, _ = self.evaluate_test_set()
            print(f"k_nearest={k}, Test Error={test_error}")

            if test_error < min_error:
                min_error = test_error
                optimal_k = k
                consecutive_no_improve = 0
                self.save_model()
            else:
                consecutive_no_improve += 1

            if consecutive_no_improve >= max_trials:
                print("Early stopping: Maximum no-improvement trials reached.")
                break

        self.k_nearest = optimal_k
        print(f"Optimal k_nearest found: {optimal_k}")
        return optimal_k

    def _cluster_data(self):
        if self.locations.shape[0] < self.n_clusters:
            print(f"Warning: Number of samples ({self.locations.shape[0]}) is less than n_clusters ({self.n_clusters}).")
            self.n_clusters = self.locations.shape[0]
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        try:
            self.cluster_labels = kmeans.fit_predict(self.locations)
            self.cluster_centers = kmeans.cluster_centers_
            return self.cluster_centers
        except:
             print(f"Error during clustering: {e}")
             return None

    def objective(self, params, dtrain, dtest):
        model = xgb.train(params, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=20, verbose_eval=False)
        y_pred = model.predict(dtest)
        y_test = dtest.get_label()
        return mean_squared_error(y_test, y_pred)
    
    def save_model(self):
        self.saved_model = self.models.copy()
        print("Model saved with k_nearest =", self.k_nearest)

    def tune_params(self, dtrain, dtest):
        search_space = {
            "max_depth": hp.choice('max_depth', range(2, 4)),
            "learning_rate": hp.uniform('learning_rate', 0.01, 0.1),
            "subsample": hp.uniform('subsample', 0.7, 1.0),
            "colsample_bytree": hp.uniform('colsample_bytree', 0.6, 1.0),
            "min_child_weight": hp.choice('min_child_weight', range(1, 5)),
            "gamma": hp.uniform('gamma', 0, 0.5),
            "reg_alpha": hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),
            "reg_lambda": hp.loguniform('reg_lambda', np.log(0.1), np.log(10.0))
        }

        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, dtrain, dtest),
            space=search_space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials
        )

        best_params = {
            'max_depth': best['max_depth'],
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': best['min_child_weight'],
            'gamma': best['gamma'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda']
        }
        return best_params

    def train_xgboost_model(self, dtrain, dtest):
        best_params = self.tune_params(dtrain, dtest)
        model = xgb.train(best_params, dtrain, num_boost_round=100)
        return model

    def fit(self, use_full_sample=None):
        """
        Fits the geographical weighted XGBoost models.

        Args:
            use_full_sample (bool | None): If True, use all samples as centers.
                                           If False, use cluster centers.
                                           If None, use the value from __init__.
        """
        # Determine the actual value for use_full_sample based on argument and instance variable
        current_use_full_sample = self.use_full_sample if use_full_sample is None else use_full_sample
        self._last_fit_use_full_sample = current_use_full_sample # Store how the last fit was done

        print(f"\nStarting model fitting with use_full_sample={current_use_full_sample} and k_nearest={self.k_nearest}...")

        centers_to_use = None
        if current_use_full_sample:
            # Use all sample locations as centers
            print(f"Using all {self.locations.shape[0]} samples as centers.")
            centers_to_use = self.locations
            self.centers_used_for_fitting = self.locations # Store which centers were used
        else:
            # Use cluster centers as centers
            # Only cluster if we haven't already or if explicitly told NOT to use full sample
            # (to ensure clustering happens if fit(False) is called first)
            if self.cluster_centers is None or (use_full_sample is not None and use_full_sample is False):
                self._cluster_data()
            
            if self.cluster_centers is None:
                raise RuntimeError("Clustering failed. Cannot fit using cluster centers.")

            print(f"Using {self.cluster_centers.shape[0]} cluster centers.")
            centers_to_use = self.cluster_centers
            self.centers_used_for_fitting = self.cluster_centers # Store which centers were used


        if centers_to_use is None or len(centers_to_use) == 0:
            raise ValueError("Centers for fitting could not be determined or are empty.")
            
        if self.k_nearest > self.data.shape[0]:
            print(f"Warning: k_nearest ({self.k_nearest}) is greater than total number of samples ({self.data.shape[0]}). Setting k_nearest to total samples.")
            effective_k_nearest = self.data.shape[0]
        else:
            effective_k_nearest = self.k_nearest

        # Build the list of indices for each center
        # This list will have a length equal to the number of centers_to_use
        # Each element is a numpy array of indices of the k_nearest neighbors to that center
        print(f"Finding {effective_k_nearest} nearest neighbors for each of the {centers_to_use.shape[0]} centers...")
        self.selected_indices_list = [
            np.argsort(np.linalg.norm(self.locations - center, axis=1))[:effective_k_nearest]
            for center in centers_to_use
        ]
        
        def train_model(i):
            nearest_indices = self.selected_indices_list[i]
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                self.data[nearest_indices], self.target[nearest_indices], nearest_indices, 
                test_size=0.3, random_state=42
            )

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            model = self.train_xgboost_model(dtrain, dtest)
            y_pred_test = model.predict(dtest)
            return model, test_indices, y_pred_test
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(train_model)(i) for i in range(len(self.selected_indices_list))
        )
        
        self.models = [result[0] for result in results]
        self.test_indices_list = [result[1] for result in results]
        self.test_predictions_list = [result[2] for result in results]


    def evaluate_test_set(self):
        if not self.test_indices_list:
           print("Warning: No test predictions available to evaluate."); 
           return None, None
        
        flat_indices = np.concatenate(self.test_indices_list)
        flat_predictions = np.concatenate(self.test_predictions_list)
        mean_preds = pd.DataFrame({'idx': flat_indices, 'pred': flat_predictions}).groupby('idx')['pred'].mean()
            
        y_true = self.target[mean_preds.index] # 直接用Series的index
        y_pred = mean_preds.values
        
        print(f"Evaluating using aggregated test-set predictions on {len(y_true)} unique samples...")
        mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
        
        return mse, r2
        
    def predict(self):
    
        flat_indices = np.concatenate(self.test_indices_list)
        flat_predictions = np.concatenate(self.test_predictions_list)
        predictions_df = pd.DataFrame({'index': flat_indices, 'predy': flat_predictions})
        predy = predictions_df.groupby('index')['predy'].mean()
        
        return predictions_df, predy

class GeoWeightedXGBoostPredictor:
    def __init__(self, models, locations, k_nearest):
        self.models = models
        self.locations = locations
        self.k_nearest = k_nearest

    def predict(self, X_new, new_locations):
        predictions = []
        for x_sample, loc in zip(X_new, new_locations):
            distances = np.linalg.norm(self.locations - loc, axis=1)
            nearest_model_idx = np.argsort(distances)[:self.k_nearest]
            model_predictions = [self.models[idx].predict(xgb.DMatrix(x_sample.reshape(1, -1))) for idx in nearest_model_idx]
            predictions.append(np.mean(model_predictions))
        return np.array(predictions)

class GeoWeightedXGBoostInterpreter:
    def __init__(self, models, use_shap=False):
        self.models = models
        self.use_shap = use_shap
        self.shap_values_df = pd.DataFrame() if use_shap else None

    def calculate_shap_values(self, model, X_test):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # print(f"SHAP values shape for current model: {shap_values.shape}")
            return shap_values
        except:
            return np.zeros(X_test.shape)

    def record_shap_values(self, shap_values, test_indices, X_test):
        shap_values_df = pd.DataFrame(
            shap_values, columns=[f"shap_{i}" for i in range(X_test.shape[1])]
        )
        shap_values_df['original_index'] = test_indices
        if self.shap_values_df.empty:
            self.shap_values_df = shap_values_df
        else:
            self.shap_values_df = pd.concat([self.shap_values_df, shap_values_df], ignore_index=True)

    def explain_all(self):
        if not self.use_shap:
            raise ValueError("SHAP explanation is disabled or no SHAP values were calculated.")
        
        if self.shap_values_df.empty:
            print("Warning: SHAP values DataFrame is empty. Ensure `calculate_shap_values` was called correctly.")
        return self.shap_values_df

class M_GWXGB_Initial:
    def __init__(self, data, target, locations, bandwidths, k_nearest=100, n_jobs=-1, use_shap=False, shap_save_dir='shap_values'):
        self.data = data
        self.target = target
        self.locations = locations
        self.bandwidths = bandwidths
        self.k_nearest = k_nearest
        self.n_jobs = n_jobs
        self.use_shap = use_shap
        self.shap_save_dir = shap_save_dir
        self.models = []
        self.optimal_bandwidths = {}

        if use_shap and not os.path.exists(shap_save_dir):
            os.makedirs(shap_save_dir)

    def fit(self):
        bandwidth_list = []
        r2_list = []
        mse_list = []
        for bandwidth in self.bandwidths:
            print(f"Training models with bandwidth: {bandwidth}")
            gwxgb_trainer = GeoWeightedXGBoostTrainer(self.data, self.target, self.locations, k_nearest=bandwidth, n_jobs=self.n_jobs)
            gwxgb_trainer.fit()
            self.models = gwxgb_trainer.models
            if self.use_shap:
                self.calculate_and_save_shap(gwxgb_trainer, bandwidth)
            bandwidth_list.append(bandwidth)
            r2_list.append(self.evaluate_test_set(gwxgb_trainer)[1])
            mse_list.append(self.evaluate_test_set(gwxgb_trainer)[0])

        print(bandwidth_list, mse_list, r2_list)

    def evaluate_test_set(self, gwxgb_trainer):
        mse_scores = []
        r2_scores = []
        test_indices_list, test_predictions_list = gwxgb_trainer.test_indices_list, gwxgb_trainer.test_predictions_list
        for i, (test_indices, y_pred_test) in enumerate(zip(test_indices_list, test_predictions_list)):
            
            y_test = self.target[test_indices]
            mse_score = mean_squared_error(y_test, y_pred_test)
            mse_scores.append(mse_score)
            r2 = r2_score(y_test, y_pred_test)
            r2_scores.append(r2)
        return np.mean(mse_scores), np.mean(r2_scores) if mse_scores else None

    def calculate_and_save_shap(self, gwxgb_trainer, bandwidth):
        test_indices_list = gwxgb_trainer.test_indices_list
        interpreter = GeoWeightedXGBoostInterpreter(models=self.models, use_shap=True)
        
        if not self.use_shap:
            return  
        
        shap_values_df = pd.DataFrame()
        for i, model in enumerate(gwxgb_trainer.models):
            test_indices = test_indices_list[i]
            X_test = self.data[test_indices]
            shap_values = interpreter.calculate_shap_values(model, X_test)
            shap_values_df_i = pd.DataFrame(shap_values, columns=[f"shap_{j}" for j in range(shap_values.shape[1])])
            shap_values_df_i['original_index'] = test_indices
            shap_values_df = pd.concat([shap_values_df, shap_values_df_i], ignore_index=True)
        shap_value_unique = shap_values_df.groupby('original_index')[[f"shap_{j}" for j in range(shap_values.shape[1])]].mean().reset_index()
        
        if self.use_shap:
            file_path_global = os.path.join(self.shap_save_dir, f'shap_values_bw{bandwidth}_global.csv')
            file_path_local = os.path.join(self.shap_save_dir, f'shap_values_bw{bandwidth}_local.csv')
            shap_value_unique.to_csv(file_path_global, index=False)
            shap_values_df.to_csv(file_path_local, index=False)
        else:
            None
            
            
class OptimizedBandwidth:
    def __init__(self, data, target, input_files, shap_save_dir='shap_values'):
        self.data = data
        self.target = target
        self.input_files = input_files
        self.shap_save_dir = shap_save_dir

    def read_columns(self, file_path, columns):
        df = pd.read_csv(file_path)
        return df[columns]

    def extract_bandwidth_from_filename(self, file_path):
        match = re.search(r'bw(\d+)', file_path)
        if match:
            return int(match.group(1))
        return None

    def optimize_bandwidths(self):

        best_combination = {} 
        y_mean = np.mean(self.target)

        columns_to_extract = [f"shap_{i}" for i in range(self.data.shape[1])]

        dataframes = {file: self.read_columns(file, columns_to_extract) for file in self.input_files}

        for col in columns_to_extract:
            best_bandwidth = None
            best_mse = float('inf')

            for file, df in dataframes.items():
                y_pred = df[col] + y_mean 
                mse = mean_squared_error(self.target, y_pred)

                bandwidth = self.extract_bandwidth_from_filename(file)

                if mse < best_mse and bandwidth is not None:
                    best_mse = mse
                    best_bandwidth = bandwidth

            best_combination[col] = best_bandwidth

        shap_values = []
        for col in columns_to_extract:
            optimal_bandwidth = best_combination[col]
            matching_file = [file for file in self.input_files if f'bw{optimal_bandwidth}' in file][0]
            shap_values.append(dataframes[matching_file][col])
        
        shap_sum = np.sum(shap_values, axis=0)
        y_pred = shap_sum + y_mean
        final_mse = mean_squared_error(self.target, y_pred)

        shap_values_df = pd.DataFrame()
        shap_values_df['y_hat'] = y_pred
        shap_values_df['target'] = self.target
        for i, col in enumerate(columns_to_extract):
            shap_values_df[col] = shap_values[i]

        file_path = os.path.join(self.shap_save_dir, 'optimized_shap_values.csv')
        shap_values_df.to_csv(file_path, index=False)

        return {'combination': best_combination, 'mse': final_mse, "bst_shap_values":shap_values_df}


class M_GXGB:
    """
    M-GWXGB implemented using a Backfitting algorithm.
    Fits base learners (GWXGB/XGB) sequentially to the
    partial residuals for each feature component. The final fitted
    learner for each component represents the estimated function f_i.
    """
    def __init__(self, data, target, locations,
                 selected_learner_types, # REQUIRED: List of 'geo' or 'xgb' per feature
                 k_range=None,
                 max_iterations=10, # Max number of full backfitting cycles (M)
                 damping_factor=0.1, # Damping (0 < damping <= 1.0). 1.0 = no damping.
                 convergence_tol_percent=1.0, # Stop if relative MSE change < this %
                 init_component_predictions=None, # Optional initial f_i(X_i)
                 xgboost_params=None,
                 enable_bw_shrinking=True, # Bandwidth shrinking (less critical for backfitting, but can keep)
                 bw_width_reduction_factor=0.8,
                 bw_stop_search_width=20,
                 bw_min_k=50,
                 n_jobs=-1,
                 ):

        self.data = data
        self.target = target.ravel()
        self.locations = locations
        self.n_samples = data.shape[0]
        self.n_features = data.shape[1]

        self.selected_learner_types = selected_learner_types
        if len(self.selected_learner_types) != self.n_features:
             raise ValueError("Length of selected_learner_types must match number of features.")

        if k_range is None:
            k_min_default = max(10, int(self.n_samples * 0.05))
            k_max_default = max(200, int(self.n_samples * 0.95))
            k_step_default = max(10, int(self.n_samples * 0.05))
            self.k_range_orig_ = [(k_min_default, k_max_default, k_step_default)] * self.n_features
        else:
            if not isinstance(k_range, list) or len(k_range) != self.n_features:
                raise ValueError("k_range must be a list of tuples, one per feature.")
            self.k_range_orig_ = k_range

        self.max_iterations = max_iterations
        self.damping_factor = damping_factor
        self.relative_tol_ = convergence_tol_percent / 100.0
        self.n_jobs = n_jobs
        self.xgboost_params = xgboost_params if xgboost_params else {}

        # Bandwidth shrinking parameters
        self.enable_bw_shrinking = enable_bw_shrinking
        self.bw_width_reduction_factor = bw_width_reduction_factor
        self.bw_stop_search_width = bw_stop_search_width
        self.bw_min_k = bw_min_k

        # --- State Variables ---
        self.intercept_ = np.mean(self.target) # Center target initially

        # Initialize component predictions f_i. 
        if init_component_predictions is not None:
            if init_component_predictions.shape == (self.n_samples, self.n_features):
                self.component_predictions_ = init_component_predictions.copy()
            else:
                raise ValueError("init_component_predictions must have shape (n_samples, n_features)")
        else:
            self.component_predictions_ = np.zeros((self.n_samples, self.n_features))

        self.current_y_pred_ = self.intercept_ + np.sum(self.component_predictions_, axis=1)
        self.local_prediction_df = pd.DataFrame()

        # Store the FINAL fitted learner instance for each component f_i
        self.fitted_learners_ = [None] * self.n_features
        self.optimal_bandwidths_ = [-1] * self.n_features
        self.geo_test_indices_ = [None] * self.n_features # For GWXGB only
        self.geo_models_ = [None] * self.n_features # For GWXGB only

        self.current_k_ranges_ = list(self.k_range_orig_)
        self.bw_search_converged_ = [False] * self.n_features

        self.history_ = {'train_mse': []} # Track training loss
        self.best_iteration_ = 0
        self.best_mse_ = np.inf

        self.best_component_predictions_ = None
        self.best_optimal_bandwidths_ = None
        self.best_fitted_learners_ = None

    # _update_dynamic_k_range remains the same as before
    def _update_dynamic_k_range(self, feature_index, current_k_range, found_k):
        """Updates the k_range for the next iteration based on the found k."""
        if not self.enable_bw_shrinking or self.bw_search_converged_[feature_index] or found_k < self.bw_min_k:
            return current_k_range

        current_min, current_max, current_step = current_k_range
        current_width = current_max - current_min
        reduction_factor = min(0.99, self.bw_width_reduction_factor)
        new_width = int(math.ceil(current_width * reduction_factor))

        if new_width < self.bw_stop_search_width or new_width <= 0:
            self.bw_search_converged_[feature_index] = True
            return current_k_range

        new_half_width = max(1, new_width // 2)
        new_min_k = max(self.bw_min_k, found_k - new_half_width)
        new_max_k = new_min_k + new_width

        if new_min_k >= new_max_k:
            self.bw_search_converged_[feature_index] = True
            return current_k_range

        next_range = (new_min_k, new_max_k, current_step)
        return next_range


    def fit(self):
        # Initial state
        last_mse = mean_squared_error(self.target, self.current_y_pred_)
        self.history_['train_mse'].append(last_mse)
        self.best_mse_ = last_mse
        self.best_iteration_ = 0
        self.best_component_predictions_ = self.component_predictions_.copy()
        self.best_optimal_bandwidths_ = list(self.optimal_bandwidths_)
        self.best_fitted_learners_ = list(self.fitted_learners_)

        # Backfitting cycles
        for m in tqdm(range(self.max_iterations), desc="Backfitting Cycles"):

            self.local_prediction_df = pd.DataFrame()
            for i in range(self.n_features):
                X_feature = self.data[:, i].reshape(-1, 1)
                learner_type = self.selected_learner_types[i]

                # --- Calculate Partial Residual for feature i ---
                # F_minus_i = F_current - f_i_current
                # Need current F: Recalculate based on current components
                current_F = self.intercept_ + np.sum(self.component_predictions_, axis=1)
                f_i_current = self.component_predictions_[:, i]
                target_h = self.target - (current_F - f_i_current)

                h_mi, fitted_trainer_instance, optimal_k_iter = None, None, -1

                try:
                    if learner_type == 'geo':
                        trainer = GeoWeightedXGBoostTrainer(
                            data=X_feature,
                            target=target_h,
                            locations=self.locations,
                            n_jobs=self.n_jobs,
                            n_clusters=100,
                        )
                        if self.bw_search_converged_[i] and self.fitted_learners_[i] is not None:
                            optimal_k_iter = self.optimal_bandwidths_[i]
                            print(f"Using converged BW k={optimal_k_iter} for feature {i}")
                            trainer.k_nearest = optimal_k_iter
                            trainer.fit()
                            local_h_mi, h_mi = trainer.predict()
                        else:
                            variable_k_range = self.current_k_ranges_[i]                           
                            optimal_k_iter = trainer.search_optimal_k_nearest(variable_k_range, max_trials=3)
                            trainer.fit()
                            print('trainer is fitted')
                            local_h_mi, h_mi = trainer.predict()
                            if not self.bw_search_converged_[i]:
                                self.current_k_ranges_[i] = self._update_dynamic_k_range(i, variable_k_range, optimal_k_iter)

                        self.optimal_bandwidths_[i] = optimal_k_iter
                        fitted_trainer_instance = trainer
                        if local_h_mi is not None:
                            self.local_prediction_df = pd.concat([self.local_prediction_df, local_h_mi], axis = 1)

                        if fitted_trainer_instance:
                           self.geo_test_indices_[i] = getattr(fitted_trainer_instance, 'test_indices_list', None)
                           self.geo_models_[i] = getattr(fitted_trainer_instance, 'models', None)

                    elif learner_type == 'xgb':
                        trainer = XGBoostTrainer_1(n_splits=5, n_jobs=self.n_jobs)
                        final_model, _, _, _, _ = trainer.calcu_oof_and_shap(
                            X_feature, target_h, tune=True, max_evals=50
                        )
                        h_mi = final_model.predict(xgb.DMatrix(X_feature))
                        local_h_mi = pd.DataFrame(h_mi)
                        self.local_prediction_df = pd.concat([self.local_prediction_df, local_h_mi], axis = 1)
                        fitted_trainer_instance = final_model
                        print(f"xgb model is fitted, the h-mi is {h_mi}")
                        self.optimal_bandwidths_[i] = 2500

                    else:
                        raise ValueError(f"Unknown learner type '{learner_type}' for feature {i}")

                    if h_mi is not None:
                        self.component_predictions_[:, i] = h_mi
                        self.fitted_learners_[i] = fitted_trainer_instance

                    else:
                        print(f"Warning: Base learner prediction failed for feature {i} in cycle {m+1}.")

                except Exception as e:
                    print(f"ERROR fitting base learner for feature {i} in cycle {m+1}: {e}.")

            # --- Recalculate Overall Prediction and MSE AFTER the full cycle ---
            self.current_y_pred_ = self.intercept_ + np.sum(self.component_predictions_, axis=1)
            current_mse = mean_squared_error(self.target, self.current_y_pred_)
            self.history_['train_mse'].append(current_mse)
            
            # --- Update best if improved ---
            if current_mse < self.best_mse_:
                self.best_mse_ = current_mse
                self.best_iteration_ = m + 1
                self.best_component_predictions_ = self.component_predictions_.copy()
                self.best_optimal_bandwidths_ = list(self.optimal_bandwidths_)
                self.best_fitted_learners_ = list(self.fitted_learners_)
                print(f" Cycle {m+1}: New best MSE={current_mse:.6f}")
            else:
                print(f" Cycle {m+1}: Train MSE={current_mse:.6f}")
                
            relative_mse_change = abs(last_mse - current_mse) / (last_mse + 1e-9)

            # Stop if relative change is too small OR patience runs out
            if relative_mse_change < self.relative_tol_ and m>0:
                print(f"\nConvergence criteria met at cycle {m+1}.")
                converged_early = True
                break

            last_mse = current_mse # Update for the next cycle's comparison

        if not converged_early:
            print(f"Termination reason: Maximum iterations ({self.max_iterations}) reached.")
        else:
            print(f"Termination reason: Converged early.")
            
        print(f"Restoring model state from BEST iteration: {self.best_iteration_} (MSE={self.best_mse_:.6f})")
        
        # Restore the best state
        self.component_predictions_ = self.best_component_predictions_
        self.optimal_bandwidths_ = self.best_optimal_bandwidths_
        self.fitted_learners_ = self.best_fitted_learners_
        
        print(f"Final training MSE: {self.best_mse_:.6f}")
        print(f"Final optimal bandwidths: {self.optimal_bandwidths_}")

    def predict(self, new_data, new_locations):
        """
        Predicts target values for new data using the final fitted component functions f_i.
        """
        if new_data.shape[1] != self.n_features:
            raise ValueError(f"Input data has {new_data.shape[1]} features, but model was trained on {self.n_features}.")
        if len(new_locations) != new_data.shape[0]:
             raise ValueError("Number of new locations must match number of new data samples.")
        if not hasattr(self, 'fitted_learners_') or self.fitted_learners_[0] is None: # Check if fitted
             raise RuntimeError("The model has not been fitted yet or fitting failed. Call fit() first.")

        # Start with the intercept
        y_pred = np.full(new_data.shape[0], self.intercept_)

        # print("Predicting using final component learners (backfitting)...")
        for i in range(self.n_features):
            final_learner_instance = self.fitted_learners_[i] # The learner representing f_i

            if final_learner_instance is None:
                print(f"Warning: No final learner stored for feature {i}. Assuming zero contribution.")
                continue 

            X_feature_new = new_data[:, i].reshape(-1, 1)
            learner_type = self.selected_learner_types[i]
            component_pred_i = np.zeros(new_data.shape[0]) # Default to zero

            try:
                if learner_type == 'geo':
                    # We need the stored *trainer* or its relevant parts (models, bandwidth)
                    geo_models_to_use = self.geo_models_[i] # Models from last fit
                    optimal_k = self.optimal_bandwidths_[i]
                    if geo_models_to_use and optimal_k > 0:

                        predictor = GeoWeightedXGBoostPredictor(geo_models_to_use, new_locations, optimal_k)
                        component_pred_i = predictor.predict(X_feature_new, new_locations)
                    else:
                        print(f"Warning: Missing models or bandwidth for Geo feature {i}. Skipping.")

                elif learner_type == 'xgb':
                    component_pred_i = final_learner_instance.predict(xgb.DMatrix(X_feature_new))

                # --- Add Component Prediction ---
                y_pred += component_pred_i

            except Exception as e:
                print(f"Error predicting component for feature {i} on new data: {e}. Skipping contribution.")

        return y_pred

    def get_optimal_bandwidths(self):
        """Returns the optimal bandwidths found for Geo components from the *last* cycle's search/fit."""
        if not hasattr(self, 'optimal_bandwidths_'):
             raise RuntimeError("The model has not been fitted yet.")
        return self.optimal_bandwidths_
        
        