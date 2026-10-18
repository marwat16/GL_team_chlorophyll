# AI treatment. ERA study 172492931 (COVID Forecasting), node 1937.
# Copied verbatim from google-research/era (docs/172492931_diffs/172492931_1937_diff.html).
# Do not edit the function body or config_list, so that the code measured is the code ERA produced.
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from typing import Any

def fit_and_predict_fn(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    config: dict[str, Any]
) -> pd.DataFrame:
    """Make probabilistic predictions for test_x by modeling train_x to train_y.

    The model uses LightGBM models for quantile regression.
    It incorporates time-series features (including lagged target variables such as y_t-1, y_t-4, etc.,
    as specified by `lag_weeks` in the config), a population-normalized and transformed
    target variable, and location information. The approach uses an iterative prediction
    strategy for the test set to correctly calculate lagged and rolling features for
    future steps, using median predictions to recursively inform future feature generation.

    This version simplifies the model by focusing exclusively on LightGBM to avoid
    potential instabilities observed with the XGBoost ensemble in previous trials.
    It maintains robust feature engineering, target transformation, and post-processing
    to ensure numerical stability, sensible predictions (e.g., non-negative, monotonic),
    and handling of missing data.

    Args:
        train_x (pd.DataFrame): Training features.
        train_y (pd.Series): Training target values (Total COVID-19 Admissions).
        test_x (pd.DataFrame): Test features (DataFrame, same features as `train_x`, but for future time periods).
        config (dict[str, Any]): Configuration parameters for the model.

    Returns:
        pd.DataFrame: DataFrame with quantile predictions for each row in test_x.
                      Columns are named 'quantile_0.XX'.
    """
    QUANTILES = [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
    ]
    TARGET_COL = 'Total COVID-19 Admissions'
    DATE_COL = 'target_end_date'
    LOCATION_COL = 'location'
    POPULATION_COL = 'population'
    HORIZON_COL = 'horizon'

    TRANSFORMED_TARGET_COL = 'transformed_admissions_per_million'

    # --- Configuration for Model and Feature Engineering ---
    # Default parameters for LightGBM
    default_lgbm_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'n_estimators': 200,
        'learning_rate': 0.03,
        'num_leaves': 25,
        'max_depth': 5,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    # Override defaults with config-specific LGBM parameters
    lgbm_params = {**default_lgbm_params, **config.get('lgbm_params', {})}

    # Feature engineering parameters.
    LAG_WEEKS = config.get('lag_weeks', [1, 4, 8, 16, 26, 52])
    LAG_DIFF_PERIODS = config.get('lag_diff_periods', [1, 2, 4, 8])
    ROLLING_WINDOWS = config.get('rolling_windows', [8, 16, 26])
    ROLLING_STD_WINDOWS = config.get('rolling_std_windows', [4, 8])

    target_transform_type = config.get('target_transform', 'fourth_root')

    # Number of LightGBM models to train for each quantile (ensemble members)
    n_lgbm_ensemble_members = config.get('n_lgbm_ensemble_members', 1)

    # Max admissions per million cap to prevent extremely large or unstable target values.
    MAX_ADMISSIONS_PER_MILLION = float(config.get('max_admissions_per_million', 6000.0))

    # Note: MIN_PRED_SPREAD is maintained in config for consistency but its role
    # in post-processing has been adjusted to focus on strict non-decreasing order after rounding,
    # rather than forcing distinct integer values with a fixed spread.
    MIN_PRED_SPREAD = float(config.get('min_pred_spread', 1))

    # --- 1. Combine train_x and train_y, and prepare for transformations ---
    df_train_full = train_x.copy()
    df_train_full[TARGET_COL] = train_y
    df_train_full[DATE_COL] = pd.to_datetime(df_train_full[DATE_COL])
    df_train_full = df_train_full.sort_values(by=[LOCATION_COL, DATE_COL]).reset_index(drop=True)

    # Determine median population from train data for robust imputation across both train and test.
    median_population_train_initial = df_train_full[POPULATION_COL].median()
    if not np.isfinite(median_population_train_initial) or median_population_train_initial == 0:
        median_population_train_initial = 1.0 # Absolute fallback for median if it's NaN/Inf/0

    # Define transform and inverse transform functions based on configuration
    if target_transform_type == 'fourth_root':
        def inverse_transform(x):
            # Ensure input to power is non-negative, and output is non-negative
            return np.maximum(0.0, np.power(np.maximum(0.0, x), 4) - 1.0)
        def forward_transform(x):
            # Ensure input to power is non-negative
            return np.power(np.maximum(0.0, x) + 1.0, 0.25)
    elif target_transform_type == 'log1p':
        def inverse_transform(x):
            # Ensure input to expm1 is non-negative and output is non-negative
            return np.expm1(np.maximum(0.0, x))
        def forward_transform(x):
            # Ensure input to log1p is non-negative
            return np.log1p(np.maximum(0.0, x))
    elif target_transform_type == 'sqrt':
        def inverse_transform(x):
            # Ensure input to power is non-negative, and output is non-negative
            return np.maximum(0.0, np.power(np.maximum(0.0, x), 2) - 1.0)
        def forward_transform(x):
            # Ensure input to sqrt is non-negative
            return np.sqrt(np.maximum(0.0, x) + 1.0)
    else: # No transformation
        def inverse_transform(x): return x
        def forward_transform(x): return x

    # Determine maximum valid transformed value based on the clipping of admissions_per_million
    MAX_TRANSFORMED_VALUE = forward_transform(MAX_ADMISSIONS_PER_MILLION)

    # --- 2. Function to add common date-based features and handle population ---
    min_date_global = df_train_full[DATE_COL].min() # Global min date for 'weeks_since_start' consistency

    def add_base_features(df_input: pd.DataFrame, min_date: pd.Timestamp, median_pop_fallback: float) -> pd.DataFrame:
        df = df_input.copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        df['year'] = df[DATE_COL].dt.year
        df['month'] = df[DATE_COL].dt.month
        df['week_of_year'] = df[DATE_COL].dt.isocalendar().week.astype(int)

        df['sin_week_of_year'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['cos_week_of_year'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        df['weeks_since_start'] = ((df[DATE_COL] - min_date).dt.days / 7).astype(int)
        df['weeks_since_start_sq'] = df['weeks_since_start']**2

        # Add horizon feature directly, as it's specific to test_x but can be used in train_x (as 0)
        if HORIZON_COL not in df.columns:
             df[HORIZON_COL] = 0 # Default for training data if not explicitly provided
        
        # Ensure population is float for all base features and handle potential NaNs/zeros
        df[POPULATION_COL] = df[POPULATION_COL].fillna(median_pop_fallback).astype(float)
        df[POPULATION_COL] = np.where(df[POPULATION_COL] == 0, 1.0, df[POPULATION_COL]) # Replace any zeros with 1.0
        
        return df

    # Apply base feature engineering to training data
    df_train_full = add_base_features(df_train_full, min_date_global, median_population_train_initial)

    # Calculate admissions per million and transform for training data
    admissions_per_million_train = df_train_full[TARGET_COL] / df_train_full[POPULATION_COL] * 1_000_000
    admissions_per_million_train = np.clip(admissions_per_million_train, 0.0, MAX_ADMISSIONS_PER_MILLION)
    df_train_full[TRANSFORMED_TARGET_COL] = forward_transform(admissions_per_million_train)
    df_train_full[TRANSFORMED_TARGET_COL] = np.clip(df_train_full[TRANSFORMED_TARGET_COL], 0.0, MAX_TRANSFORMED_VALUE)

    # Prepare test_x
    test_x_processed = add_base_features(test_x.copy(), min_date_global, median_population_train_initial)


    BASE_FEATURES = [POPULATION_COL, 'year', 'month', 'week_of_year',
                     'sin_week_of_year', 'cos_week_of_year', 'weeks_since_start',
                     'weeks_since_start_sq', HORIZON_COL]

    CATEGORICAL_FEATURES_LIST = [LOCATION_COL]

    # --- 3. Generate time-series dependent features for training data ---
    train_features_df = df_train_full.copy()

    # Pre-calculate fallback for missing history. This will be median of the *transformed* target.
    median_transformed_train_y_fallback = df_train_full[TRANSFORMED_TARGET_COL].median()
    if not np.isfinite(median_transformed_train_y_fallback) or df_train_full[TRANSFORMED_TARGET_COL].empty:
        median_transformed_train_y_fallback = forward_transform(1.0) # Corresponds to 1 admission per million after inverse transform
    median_transformed_train_y_fallback = np.clip(median_transformed_train_y_fallback, 0.0, MAX_TRANSFORMED_VALUE)


    for lag in LAG_WEEKS:
        train_features_df[f'lag_{lag}_wk'] = train_features_df.groupby(LOCATION_COL)[TRANSFORMED_TARGET_COL].shift(lag)

    for diff_period in LAG_DIFF_PERIODS:
        train_features_df[f'diff_lag_1_period_{diff_period}_wk'] = \
            train_features_df.groupby(LOCATION_COL)[TRANSFORMED_TARGET_COL].diff(periods=diff_period).shift(1)

    for window in ROLLING_WINDOWS:
        train_features_df[f'rolling_mean_{window}_wk'] = \
            train_features_df.groupby(LOCATION_COL)[TRANSFORMED_TARGET_COL].transform(
                lambda x: x.rolling(window=window, min_periods=1, closed='left').mean()
            )

    for window in ROLLING_STD_WINDOWS:
        train_features_df[f'rolling_std_{window}_wk'] = \
            train_features_df.groupby(LOCATION_COL)[TRANSFORMED_TARGET_COL].transform(
                lambda x: x.rolling(window=window, min_periods=1, closed='left').std()
            )

    y_train_model = train_features_df[TRANSFORMED_TARGET_COL]

    train_specific_features = [f'lag_{lag}_wk' for lag in LAG_WEEKS] + \
                              [f'diff_lag_1_period_{p}_wk' for p in LAG_DIFF_PERIODS] + \
                              [f'rolling_mean_{window}_wk' for window in ROLLING_WINDOWS] + \
                              [f'rolling_std_{window}_wk' for window in ROLLING_STD_WINDOWS]

    X_train_model = train_features_df[BASE_FEATURES + CATEGORICAL_FEATURES_LIST + train_specific_features].copy()

    # Handle missing data introduced by lagging/rolling in training features
    for col in train_specific_features:
        if X_train_model[col].isnull().any():
            # Use ffill/bfill within groups, then fallback to a default if still NaN (e.g. for new locations at start)
            X_train_model[col] = X_train_model.groupby(LOCATION_COL)[col].transform(lambda x: x.ffill().bfill())
            fill_value = 0.0 if 'std' in col or 'diff' in col else median_transformed_train_y_fallback
            X_train_model[col] = X_train_model[col].fillna(fill_value)


    # Drop rows where the target itself is NaN (shouldn't happen with valid train_y from harness)
    train_combined = X_train_model.copy()
    train_combined[TRANSFORMED_TARGET_COL] = y_train_model
    train_combined.dropna(subset=[TRANSFORMED_TARGET_COL], inplace=True)

    X_train_model = train_combined.drop(columns=[TRANSFORMED_TARGET_COL])
    y_train_model = train_combined[TRANSFORMED_TARGET_COL]

    # Handle cases where training data becomes empty after processing (e.g., very early folds)
    if X_train_model.empty or y_train_model.empty:
        print("Warning: Training data is empty after preprocessing. Returning fallback predictions (all zeros).")
        predictions_df = pd.DataFrame(index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES])
        for q_col in [f'quantile_{q}' for q in QUANTILES]:
            predictions_df[q_col] = 0
        return predictions_df


    # Handle categorical features for LightGBM.
    # Ensure all possible location categories (from both train and test) are known for categorical dtype.
    all_location_categories = pd.unique(pd.concat([X_train_model[LOCATION_COL], test_x_processed[LOCATION_COL]]))
    all_location_categories_list = all_location_categories.tolist()

    # Prepare X_train for LightGBM (categorical Dtype)
    X_train_lgbm = X_train_model.copy()
    X_train_lgbm[LOCATION_COL] = X_train_lgbm[LOCATION_COL].astype(pd.CategoricalDtype(categories=all_location_categories_list))

    # Store the final column order from training data to ensure consistency during prediction
    X_train_model_cols = X_train_model.columns.tolist()
    categorical_feature_names = [col for col in CATEGORICAL_FEATURES_LIST if col in X_train_model_cols]

    # --- 4. Model Training (Only LightGBM models) ---
    models = {q: [] for q in QUANTILES} # List of models for each quantile

    for q in QUANTILES:
        for i in range(n_lgbm_ensemble_members):
            lgbm_model_params_i = lgbm_params.copy()
            lgbm_model_params_i['alpha'] = q
            lgbm_model_params_i['random_state'] = lgbm_model_params_i['random_state'] + i

            lgbm_model = LGBMRegressor(**lgbm_model_params_i)
            lgbm_model.fit(X_train_lgbm, y_train_model,
                           categorical_feature=categorical_feature_names)
            models[q].append(lgbm_model)

    # --- 5. Iterative Feature Generation and Prediction for Test Data ---
    # `location_history_data` stores transformed target values for each location to generate lags.
    # Initialize with training data's transformed target values
    location_history_data = df_train_full.groupby(LOCATION_COL)[TRANSFORMED_TARGET_COL].apply(list).to_dict()

    original_test_x_index = test_x.index

    # Sort test_x to ensure chronological processing within each location for iterative predictions
    test_x_processed['original_index'] = test_x_processed.index
    test_x_processed[DATE_COL] = pd.to_datetime(test_x_processed[DATE_COL])
    test_x_processed = test_x_processed.sort_values(by=[LOCATION_COL, DATE_COL]).reset_index(drop=True)

    predictions_df = pd.DataFrame(index=original_test_x_index, columns=[f'quantile_{q}' for q in QUANTILES])

    for idx, row in test_x_processed.iterrows():
        current_loc = row[LOCATION_COL]
        original_idx = row['original_index']

        current_loc_hist = location_history_data.get(current_loc, [])

        # Populate current features dictionary, population is already cleaned by add_base_features
        current_features_dict = {col: row[col] for col in BASE_FEATURES}

        # Generate time-series features for the current test row using available history
        if not current_loc_hist:
            # If no history, use fallbacks for all time-series features
            for lag in LAG_WEEKS:
                current_features_dict[f'lag_{lag}_wk'] = median_transformed_train_y_fallback
            for diff_period in LAG_DIFF_PERIODS:
                current_features_dict[f'diff_lag_1_period_{diff_period}_wk'] = 0.0
            for window in ROLLING_WINDOWS:
                current_features_dict[f'rolling_mean_{window}_wk'] = median_transformed_train_y_fallback
            for window in ROLLING_STD_WINDOWS:
                current_features_dict[f'rolling_std_{window}_wk'] = 0.0
        else:
            # Generate features based on available history
            for lag in LAG_WEEKS:
                lag_col_name = f'lag_{lag}_wk'
                if len(current_loc_hist) >= lag:
                    lag_value = current_loc_hist[-lag]
                else:
                    lag_value = median_transformed_train_y_fallback
                current_features_dict[lag_col_name] = lag_value

            for diff_period in LAG_DIFF_PERIODS:
                diff_col_name = f'diff_lag_1_period_{diff_period}_wk'
                # Use current_loc_hist[-1] as the reference point for differences
                if len(current_loc_hist) >= diff_period + 1:
                    diff_value = current_loc_hist[-1] - current_loc_hist[-(diff_period + 1)]
                elif len(current_loc_hist) >= 2: # Fallback to 1-period diff if longer diff not possible
                    diff_value = current_loc_hist[-1] - current_loc_hist[-2]
                else:
                    diff_value = 0.0 # If not enough history for any diff
                current_features_dict[diff_col_name] = diff_value

            for window in ROLLING_WINDOWS:
                rolling_col_name = f'rolling_mean_{window}_wk'
                actual_window_size = min(window, len(current_loc_hist))
                if actual_window_size > 0:
                    rolling_mean_val = np.mean(current_loc_hist[-actual_window_size:])
                else:
                    rolling_mean_val = median_transformed_train_y_fallback
                current_features_dict[rolling_col_name] = rolling_mean_val

            for window in ROLLING_STD_WINDOWS:
                rolling_std_col_name = f'rolling_std_{window}_wk'
                actual_window_size = min(window, len(current_loc_hist))
                if actual_window_size > 1: # Need at least 2 points for std dev
                    rolling_std_val = np.std(current_loc_hist[-actual_window_size:])
                else:
                    rolling_std_val = 0.0
                current_features_dict[rolling_std_col_name] = rolling_std_val

        # Create a DataFrame for the current row's features
        X_test_row_base = pd.DataFrame([current_features_dict])

        # Reindex to ensure all columns from training set are present and in order.
        X_test_row_base = X_test_row_base.reindex(columns=X_train_model_cols, fill_value=0.0)

        # Prepare X_test_row for LGBM (categorical Dtype)
        X_test_row_lgbm = X_test_row_base.copy()
        X_test_row_lgbm[LOCATION_COL] = X_test_row_lgbm[LOCATION_COL].astype(pd.CategoricalDtype(categories=all_location_categories_list))

        row_predictions_transformed = {}
        for q in QUANTILES:
            ensemble_preds_for_q = []
            for lgbm_model_q in models[q]:
                try:
                    pred = lgbm_model_q.predict(X_test_row_lgbm)[0]
                    if np.isfinite(pred):
                        pred_clipped = np.clip(pred, 0.0, MAX_TRANSFORMED_VALUE)
                        ensemble_preds_for_q.append(pred_clipped)
                except Exception:
                    # Silently fail for individual model prediction to allow ensemble to continue
                    pass 

            if ensemble_preds_for_q:
                row_predictions_transformed[q] = np.median(ensemble_preds_for_q) # Use median for robustness
            else:
                row_predictions_transformed[q] = median_transformed_train_y_fallback

        transformed_preds_array = np.array([row_predictions_transformed[q] for q in QUANTILES])
        transformed_preds_array = np.clip(transformed_preds_array, 0.0, MAX_TRANSFORMED_VALUE)

        inv_preds_admissions_per_million = inverse_transform(transformed_preds_array)
        inv_preds_admissions_per_million = np.clip(inv_preds_admissions_per_million, 0.0, MAX_ADMISSIONS_PER_MILLION)

        population_val = current_features_dict[POPULATION_COL]
        # Calculate total admissions as float before final post-processing and rounding
        final_preds_total_admissions = inv_preds_admissions_per_million * population_val / 1_000_000

        for i, q in enumerate(QUANTILES):
            predictions_df.loc[original_idx, f'quantile_{q}'] = final_preds_total_admissions[i]

        # Use the median prediction from the current forecasts to update the history for next iteration
        # This value is based on the *transformed* prediction (before inverse transform and scaling)
        median_pred_transformed_raw = row_predictions_transformed[0.5]
        median_pred_transformed_clipped = np.clip(median_pred_transformed_raw, 0.0, MAX_TRANSFORMED_VALUE)
        
        # If the median prediction is NaN/Inf, use a robust fallback for history
        if not np.isfinite(median_pred_transformed_clipped):
             value_to_add_to_history = median_transformed_train_y_fallback
        else:
             value_to_add_to_history = median_pred_transformed_clipped

        location_history_data.setdefault(current_loc, []).append(value_to_add_to_history)


    # --- 6. Post-processing to ensure non-negativity, integer values, and non-decreasing monotonicity ---
    predictions_array = predictions_df.values.astype(float) # Get predictions as floats

    # Apply non-negativity and rounding to integers for each prediction first
    predictions_array = np.round(np.maximum(0, predictions_array)).astype(int)

    for i in range(predictions_array.shape[0]):
        # Sort each row (quantiles) to ensure non-decreasing monotonicity
        # This means Q_j >= Q_{j-1}. If multiple quantiles predict the same integer value,
        # they will remain the same, allowing for zero-width intervals. This is consistent
        # with the typical statistical definition of 'monotonically increasing' (non-decreasing).
        predictions_array[i, :] = np.sort(predictions_array[i, :])
        
    predictions_df = pd.DataFrame(predictions_array, columns=predictions_df.columns, index=predictions_df.index)
    # Ensure final output DataFrame has integer type columns as required for count data.
    predictions_df = predictions_df.astype(int)

    return predictions_df

# YOUR config_list
config_list = [
    { # Config 1: Optimized base (similar to previous best with slight adjustments)
        'lgbm_params': {
            'n_estimators': 280,
            'learning_rate': 0.028,
            'num_leaves': 26,
            'max_depth': 5,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.15, # Slight increase in L1 regularization
            'reg_lambda': 0.15 # Slight increase in L2 regularization
        },
        'target_transform': 'fourth_root',
        'lag_weeks': [1, 4, 8, 16, 26, 52],
        'lag_diff_periods': [1, 2, 4, 8],
        'rolling_windows': [8, 16, 26],
        'rolling_std_windows': [4, 8],
        'n_lgbm_ensemble_members': 1, # Using only 1 LGBM per quantile
        'max_admissions_per_million': 5000.0,
        'min_pred_spread': 1 # This parameter's role is re-evaluated in the new post-processing
    },
    { # Config 2: More Aggressive Regularization
        'lgbm_params': {
            'n_estimators': 350, # More trees to compensate for smaller LR
            'learning_rate': 0.02, # Reduced learning rate
            'num_leaves': 20, # Reduced complexity per tree
            'max_depth': 4, # Reduced max depth per tree
            'min_child_samples': 30, # Increased minimum samples per leaf
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'colsample_bytree': 0.7, # Slightly more feature subsampling
            'subsample': 0.7, # Slightly more data subsampling
            'reg_alpha': 0.5, # Stronger L1 regularization
            'reg_lambda': 0.5 # Stronger L2 regularization
        },
        'target_transform': 'fourth_root',
        'lag_weeks': [1, 4, 8, 16, 26, 52],
        'lag_diff_periods': [1, 2, 4, 8],
        'rolling_windows': [8, 16, 26],
        'rolling_std_windows': [4, 8],
        'n_lgbm_ensemble_members': 1,
        'max_admissions_per_million': 5000.0,
        'min_pred_spread': 1
    }
]