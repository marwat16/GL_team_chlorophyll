from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

#Child #851
# The solution aims to improve accuracy and generalize effectively across diverse GiftEval datasets
# by refining the fit_and_predict_fn and its configurations.
MODEL_NAME = "HybridDecompositionModel" # Name your solution here
 
MODEL_VERSION = 9 # Incremented version to reflect latest review and validation
# The config_list offers a diverse set of forecasting strategies, balancing between
# robustness, adaptability, and the ability to capture complex patterns (trend, seasonality, residuals).
# The goal is to maximize performance across GiftEval datasets within the `MAX_CONFIGS` limit of 8,
# by ensuring each configuration is distinct and powerful.
config_list = [
    # Config 0: Last Value Naive. Simple and ultra-robust, suitable for highly erratic or very short series.
    {'name': 'last_value_naive_0', 'components': [{'type': 'base_level', 'base_level_method': 'last_value'}], 'non_negative': False, 'transform_log': False, 'version': MODEL_VERSION},
    # Config 1: Multiplicative: Log-Linear Trend + Primary Seasonal. Effective for exponential growth/decay with strong primary seasonality.
    {'name': 'multiplicative_linear_trend_primary_seasonal_log_1', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 30},
        {'type': 'trend', 'trend_method': 'linear_polyfit', 'trend_degree': 1, 'trend_window_multiplier': 50.0}, # Localized linear trend
        {'type': 'seasonal_primary', 'seasonal_avg_window_multiplier': 2.5}
    ], 'non_negative': True, 'transform_log': True, 'version': MODEL_VERSION},
    # Config 2: Additive: More Reactive Linear Trend + Primary Seasonal + Median RC. For moderately fast-changing patterns with clear primary seasonality.
    {'name': 'additive_more_reactive_trend_seasonal_rc_2', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 30}, # More reactive base
        {'type': 'trend', 'trend_method': 'linear_polyfit', 'trend_degree': 1, 'trend_window_multiplier': 20.0}, # Smoother reactive linear trend
        {'type': 'seasonal_primary', 'seasonal_avg_window_multiplier': 3.0},
        {'type': 'residual_correction', 'rc_window_size': 15, 'residual_correction_method': 'median', 'residual_correction_decay_enabled': True, 'residual_correction_damping_factor': 0.99} # Increased damping factor from 0.98 to 0.99 for more persistence
    ], 'non_negative': False, 'transform_log': False, 'version': MODEL_VERSION},
    # Config 3: Multiplicative: Log-Constant Trend + Secondary Seasonal (DOW/HOD) + RC. Ideal for high-frequency data where complex daily/weekly patterns scale.
    {'name': 'multiplicative_const_trend_secondary_seasonal_rc_log_3', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 60},
        {'type': 'trend', 'trend_method': 'constant_median_robust', 'trend_window_multiplier': None},
        {'type': 'seasonal_secondary', 'use_dayofweek': True, 'use_hourofday': True, 'use_month_of_year': False, 'use_day_of_year': False}, # Focus on high freq secondary
        {'type': 'residual_correction', 'rc_window_size': 30, 'residual_correction_method': 'median', 'residual_correction_decay_enabled': True, 'residual_correction_damping_factor': 0.99} # Reduced damping factor from 1.0 to 0.99 for slight decay
    ], 'non_negative': True, 'transform_log': True, 'version': MODEL_VERSION},
    # Config 4: Additive: Adaptive Stable Base + Long Primary Seasonal Only. For very stable, strong seasonal patterns with minimal trend/noise.
    {'name': 'additive_stable_base_long_primary_seasonal_4', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 200}, # Adaptive stable base
        {'type': 'seasonal_primary', 'seasonal_avg_window_multiplier': 15.0} # Longer averaging window for seasonality
    ], 'non_negative': False, 'transform_log': False, 'version': MODEL_VERSION},
    # Config 5: Additive: Long-term Linear Trend + Primary Seasonal. Robust for datasets with consistent, global linear changes.
    {'name': 'additive_long_term_linear_trend_primary_5', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 50},
        {'type': 'trend', 'trend_method': 'linear_polyfit', 'trend_degree': 1, 'trend_window_multiplier': None}, # Use full history for trend
        {'type': 'seasonal_primary', 'seasonal_avg_window_multiplier': 5.0} # Moderately long seasonal average
    ], 'non_negative': False, 'transform_log': False, 'version': MODEL_VERSION},
    # Config 6: Multiplicative: Log-Constant Trend + Primary Seasonal + RC. For log-transformed series with a stable level, but seasonality/residuals are key.
    {'name': 'multiplicative_stable_primary_rc_6_log', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 60},
        {'type': 'trend', 'trend_method': 'constant_median_robust', 'trend_window_multiplier': None},
        {'type': 'seasonal_primary', 'seasonal_avg_window_multiplier': 8.0},
        {'type': 'residual_correction', 'rc_window_size': 20, 'residual_correction_method': 'median', 'residual_correction_decay_enabled': True, 'residual_correction_damping_factor': 0.98} # Increased damping factor from 0.95 to 0.98 for more persistence
    ], 'non_negative': True, 'transform_log': True, 'version': MODEL_VERSION},
    # Config 7: Comprehensive Additive: Linear Trend + Full Secondary Seasonal + RC. Robust for complex datasets with multiple seasonalities and a general trend.
    {'name': 'comprehensive_additive_seasonal_rc_7', 'components': [
        {'type': 'base_level', 'base_level_method': 'median_last_k_window', 'k_window_size': 30}, # More reactive base
        {'type': 'trend', 'trend_method': 'linear_polyfit', 'trend_degree': 1, 'trend_window_multiplier': None},
        {'type': 'seasonal_secondary', 'use_dayofweek': True, 'use_hourofday': True, 'use_month_of_year': True, 'use_day_of_year': True},
        {'type': 'residual_correction', 'rc_window_size': 25, 'residual_correction_method': 'median', 'residual_correction_decay_enabled': True, 'residual_correction_damping_factor': 0.90} # Increased damping factor from 0.85 to 0.90 for less aggressive decay
    ], 'non_negative': True, 'transform_log': False, 'version': MODEL_VERSION},
]
# Helper function for robust trend calculation
def _calculate_trend_and_fitted(
    historical_residuals: np.ndarray,
    prediction_length: int,
    train_len: int,
 
    season_length: int, # Used for trend window multiplier, can be 1 for low freq
    trend_method: str,
    trend_degree: int, # Only relevant for 'linear_polyfit'
    trend_window_multiplier: Any # Can be float or None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates trend forecast and fitted trend on historical data.
    Returns (trend_forecast_component, fitted_trend_on_full_train).
    Ensures outputs are finite by falling back to zero if trend cannot be reliably estimated.
    Trend is projected over the entire training history for consistent residual calculation.
    """
    trend_forecast_component = np.zeros(prediction_length, dtype=float)
    fitted_trend_on_full_train = np.zeros(train_len, dtype=float)
    if train_len < 1:
        return trend_forecast_component, fitted_trend_on_full_train
    # Determine the segment of data to use for fitting the trend
    window_size = train_len # Default to full history (no windowing)
    if trend_window_multiplier is not None:
        multiplier = trend_window_multiplier
 
        if season_length > 0: # Use provided season_length for dynamic windowing if available
            window_size_base = int(multiplier * season_length)
 
        else: # Fallback if season_length is 0 or less
            window_size_base = int(multiplier)
        
        window_size = max(1, min(window_size_base, train_len))
    else:
        window_size = train_len # Use full history
    fit_start_idx = train_len - window_size
    y_fit_segment = historical_residuals[fit_start_idx:]
    # X-coordinates for fitting are the actual absolute indices within the training data
    x_fit_segment_raw = np.arange(fit_start_idx, train_len)
    if len(y_fit_segment) < 1 or len(x_fit_segment_raw) < 1: # No data in segment to fit
        return trend_forecast_component, fitted_trend_on_full_train
    # If degree is 0, or method is 'constant_median_robust', use median for robustness
    if trend_degree == 0 or trend_method == 'constant_median_robust':
        median_val = np.median(y_fit_segment)
        trend_forecast_component = np.full(prediction_length, median_val)
        fitted_trend_on_full_train = np.full(train_len, median_val) # Applies uniformly
    elif trend_method == 'linear_polyfit':
        # Add explicit check for constant segment to prevent polyfit errors/instability
        # Use np.isclose for floating-point robustness
        if np.all(np.isclose(y_fit_segment, y_fit_segment[0], atol=1e-9)):
            median_val = y_fit_segment[0]
            trend_forecast_component = np.full(prediction_length, median_val)
            fitted_trend_on_full_train = np.full(train_len, median_val) # Applies uniformly
            return trend_forecast_component, fitted_trend_on_full_train
        effective_degree = min(trend_degree, len(y_fit_segment) - 1)
        effective_degree = max(0, effective_degree) # Ensure degree is non-negative
        # Fallback to constant median if not enough points for polynomial degree
        if len(y_fit_segment) <= effective_degree:
            median_val = np.median(y_fit_segment)
            trend_forecast_component = np.full(prediction_length, median_val)
            fitted_trend_on_full_train = np.full(train_len, median_val)
            return trend_forecast_component, fitted_trend_on_full_train
        try:
            # Center x-values around their mean to improve numerical stability of polyfit
            x_mean = np.mean(x_fit_segment_raw)
            x_fit_segment_centered = x_fit_segment_raw - x_mean
            # Fit polynomial using the centered x-coordinates
            poly_coeffs = np.polyfit(x_fit_segment_centered, y_fit_segment, effective_degree)
            trend_poly = np.poly1d(poly_coeffs)
            # Forecast: x-values for prediction need to be absolute indices extending from train_len, then centered
            x_forecast_raw = np.arange(train_len, train_len + prediction_length)
            x_forecast_centered = x_forecast_raw - x_mean
            trend_forecast_component = trend_poly(x_forecast_centered)
            # Fitted: For the entire training history, evaluate the trend polynomial for each point
            # using their absolute indices, then centered. This ensures consistent detrending.
            x_full_train_raw = np.arange(train_len)
            x_full_train_centered = x_full_train_raw - x_mean
            fitted_trend_on_full_train = trend_poly(x_full_train_centered)
        except np.linalg.LinAlgError:
            # Fallback to constant median if polyfit fails (e.g., singular matrix)
            median_val = np.median(y_fit_segment)
            trend_forecast_component = np.full(prediction_length, median_val)
            fitted_trend_on_full_train = np.full(train_len, median_val) # Applies uniformly
    
    # Ensure outputs are finite before returning as a final safeguard against extreme values from polyfit.
    return np.nan_to_num(trend_forecast_component, nan=0.0, posinf=0.0, neginf=0.0), \
           np.nan_to_num(fitted_trend_on_full_train, nan=0.0, posinf=0.0, neginf=0.0)
# --- New Modular Component Functions ---
def _fit_predict_base_level(
    historical_data: np.ndarray, prediction_length: int, train_len: int, season_length: int, component_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a dynamic base level based on `base_level_method` and returns it as forecast and fitted.
    `historical_data` is expected to be finite (e.g., `processed_targets_np` or its log-transformed version).
    `season_length` is used for 'median_last_season' and default 'k_window_size'.
    """
    if train_len == 0:
        return np.zeros(prediction_length), np.zeros(train_len)
    base_level_method = component_config.get('base_level_method', 'median_all_history')
    base_val = 0.0 # Default fallback
    if base_level_method == 'last_value':
        base_val = historical_data[-1]
    elif base_level_method == 'median_last_season':
        effective_season_length = max(1, season_length)
        if train_len >= effective_season_length:
            base_val = np.median(historical_data[-effective_season_length:])
        else:
            base_val = np.median(historical_data)
    elif base_level_method == 'median_last_k_window':
 
        k_window_size_default = max(7, season_length) if season_length > 0 else 7 # Use season_length for default
        k_window_size_val = component_config.get('k_window_size', k_window_size_default)
        k_window_size_val = max(1, min(k_window_size_val, train_len)) # Ensure valid window size
        base_val = np.median(historical_data[-k_window_size_val:])
    elif base_level_method == 'zero_constant':
        base_val = 0.0
    else: # Default or 'median_all_history'
        base_val = np.median(historical_data)
    return np.full(prediction_length, base_val), np.full(train_len, base_val)
def _fit_predict_trend_component(
    historical_data: np.ndarray, prediction_length: int, train_len: int, season_length: int, component_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
 
    """Wrapper for _calculate_trend_and_fitted. historical_data is guaranteed finite.
    `season_length` is passed through to `_calculate_trend_and_fitted` for windowing."""
    if train_len < 1:
        return np.zeros(prediction_length), np.zeros(train_len)
    return _calculate_trend_and_fitted(
        historical_residuals=historical_data, # This input is guaranteed finite
        prediction_length=prediction_length,
        train_len=train_len,
 
        season_length=season_length, # Pass original season_length for trend window context
        trend_method=component_config.get('trend_method', 'linear_polyfit'),
        trend_degree=component_config.get('trend_degree', 1),
        trend_window_multiplier=component_config.get('trend_window_multiplier')
    )
# New helper functions for frequency type checks
def _dataset_has_hourly_or_subhourly_freq(freq_str: str) -> bool:
    """True if frequency is hourly or sub-hourly."""
    return any(f in freq_str for f in ['T', 'H'])
def _dataset_has_daily_or_coarser_freq(freq_str: str) -> bool:
    """True if frequency is daily or coarser (weekly, monthly, etc.)."""
    return any(f in freq_str for f in ['D', 'W', 'M', 'Q', 'A'])
def _fit_predict_seasonal_primary_component(
    historical_data: np.ndarray, prediction_length: int, train_len: int, season_length: int, component_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates primary seasonal pattern based on `season_length`.
    `historical_data` is expected to be finite.
    `season_length` here is the *effective* one determined by fit_and_predict_fn.
    """
    seasonal_forecast_component = np.zeros(prediction_length, dtype=float)
    fitted_seasonal_on_train = np.zeros(train_len, dtype=float)
 
    effective_season_length = max(1, season_length) # Ensure season_length is at least 1
    if effective_season_length <= 1 or train_len == 0:
        return seasonal_forecast_component, fitted_seasonal_on_train
    seasonal_avg_window_multiplier = component_config.get('seasonal_avg_window_multiplier', 3.0)
    seasonal_data_len_for_avg = min(train_len, int(seasonal_avg_window_multiplier * effective_season_length))
    seasonal_pattern = np.zeros(effective_season_length, dtype=float)
    if seasonal_data_len_for_avg > 0:
        seasonal_avg_data = historical_data[-seasonal_data_len_for_avg:]
        # Pad with NaNs for robust median calculation across cycles. Padding at the beginning.
        # This ensures that np.nanmedian can operate on full columns, even if the last cycle is incomplete.
        padding_needed = (effective_season_length - (len(seasonal_avg_data) % effective_season_length)) % effective_season_length
        padded_seasonal_avg_data = np.pad(seasonal_avg_data, (padding_needed, 0), 'constant', constant_values=np.nan)
        num_cycles = len(padded_seasonal_avg_data) // effective_season_length
        # Only compute seasonal pattern if at least one full cycle is available
        if num_cycles > 0:
            reshaped_data = padded_seasonal_avg_data.reshape(num_cycles, effective_season_length)
            seasonal_pattern = np.nanmedian(reshaped_data, axis=0) # Use np.nanmedian to handle NaNs from padding
    
    # Ensure seasonal pattern is finite before tiling and using in predictions.
    # np.nanmedian can return NaN if all values in a column are NaN.
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=0.0, posinf=0.0, neginf=0.0)
    effective_pattern_length = max(1, len(seasonal_pattern))
    num_repeats_seasonal = (prediction_length + effective_pattern_length - 1) // effective_pattern_length
    seasonal_forecast_component = np.tile(seasonal_pattern, num_repeats_seasonal)[:prediction_length]
    fitted_seasonal_on_train = np.tile(seasonal_pattern, (train_len + effective_pattern_length - 1) // effective_pattern_length)[:train_len]
    return seasonal_forecast_component, fitted_seasonal_on_train
def _fit_predict_seasonal_secondary_component(
    historical_data: np.ndarray, prediction_index: pd.Index, input_targets_index: pd.Index,
    prediction_length: int, train_len: int, component_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates secondary seasonal patterns (DOW, HOD, Month of Year, Day of Year) from residuals.
    `historical_data` is expected to be finite.
    Optimized to use pure NumPy for median calculations, avoiding Pandas groupby overhead.
    Includes logic for DayOfWeek-HourOfDay interaction.
    """
    secondary_seasonal_forecast_component = np.zeros(prediction_length, dtype=float)
    fitted_secondary_seasonal_on_train = np.zeros(train_len, dtype=float)
    fallback_value = 0.0 # Sensible fallback for additive components
    freq_str = input_targets_index.freqstr if input_targets_index.freqstr is not None else ''
    if not freq_str or train_len == 0:
        return secondary_seasonal_forecast_component, fitted_secondary_seasonal_on_train
    is_hourly_or_subhourly = _dataset_has_hourly_or_subhourly_freq(freq_str)
    is_daily_or_coarser = _dataset_has_daily_or_coarser_freq(freq_str)
    data_span_days = (input_targets_index.max() - input_targets_index.min()).days if train_len > 1 else 0
    min_years_for_yearly_seasonality = 2.0
    # Get datetime attributes for training and prediction indices
    train_dayofweek = input_targets_index.dayofweek.values
    pred_dayofweek = prediction_index.dayofweek.values
    train_hour = input_targets_index.hour.values
    pred_hour = prediction_index.hour.values
    train_month = input_targets_index.month.values
    pred_month = prediction_index.month.values
    train_dayofyear = input_targets_index.dayofyear.values
    pred_dayofyear = prediction_index.dayofyear.values
    # Helper to calculate median for a given attribute
    def _calculate_seasonal_median(attribute_values_train: np.ndarray, attribute_values_predict: np.ndarray, num_possible_values: int) -> Tuple[np.ndarray, np.ndarray]:
        mapping_array = np.full(num_possible_values, fallback_value, dtype=float)
        
        # Identify unique attribute values present in training data
        unique_attrs_train = np.unique(attribute_values_train)
        for attr_val in unique_attrs_train:
            mask = (attribute_values_train == attr_val)
            # Ensure there's data for this attribute value and it's finite
            valid_residuals = historical_data[mask]
            if len(valid_residuals) > 0 and np.any(np.isfinite(valid_residuals)):
                mapping_array[attr_val] = np.median(valid_residuals)
        
        # Ensure mapping array is finite
        mapping_array = np.nan_to_num(mapping_array, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)
        
        # Apply mapping to training and prediction sets
        fitted_comp = mapping_array[attribute_values_train]
        forecast_comp = mapping_array[attribute_values_predict]
        
        return forecast_comp, fitted_comp
    # Determine which components are requested by the config
    do_dow_config = component_config.get('use_dayofweek', False)
    do_hod_config = component_config.get('use_hourofday', False)
    do_month_config = component_config.get('use_month_of_year', False)
    do_doy_config = component_config.get('use_day_of_year', False)
    # Handle DayOfWeek-HourOfDay interaction if both are true and frequency is suitable
    if do_dow_config and do_hod_config and is_hourly_or_subhourly:
        # Create composite feature (0-167 for DayOfWeek*24 + Hour)
        train_composite_dh = train_dayofweek * 24 + train_hour
        pred_composite_dh = pred_dayofweek * 24 + pred_hour
        num_dh_values = 7 * 24 # Total possible combinations
        forecast_comp_dh, fitted_comp_dh = _calculate_seasonal_median(train_composite_dh, pred_composite_dh, num_dh_values)
        secondary_seasonal_forecast_component += forecast_comp_dh
        fitted_secondary_seasonal_on_train += fitted_comp_dh
        # Disable individual DOW and HOD so they are not added again
        do_dow_config = False
        do_hod_config = False
    # Add individual Day of Week component if not covered by interaction
    if do_dow_config and (is_hourly_or_subhourly or is_daily_or_coarser):
        forecast_comp, fitted_comp = _calculate_seasonal_median(train_dayofweek, pred_dayofweek, 7) # 0-6 for dayofweek
        secondary_seasonal_forecast_component += forecast_comp
        fitted_secondary_seasonal_on_train += fitted_comp
    # Add individual Hour of Day component if not covered by interaction
    if do_hod_config and is_hourly_or_subhourly:
        forecast_comp, fitted_comp = _calculate_seasonal_median(train_hour, pred_hour, 24) # 0-23 for hour
        secondary_seasonal_forecast_component += forecast_comp
        fitted_secondary_seasonal_on_train += fitted_comp
    # Add Month of Year component
    if do_month_config and \
       is_daily_or_coarser and \
       data_span_days >= (365 * min_years_for_yearly_seasonality):
        forecast_comp, fitted_comp = _calculate_seasonal_median(train_month - 1, pred_month - 1, 12) # Months 1-12, map to 0-11
        secondary_seasonal_forecast_component += forecast_comp
        fitted_secondary_seasonal_on_train += fitted_comp
    # Add Day of Year component
    if do_doy_config and \
       is_daily_or_coarser and \
       data_span_days >= (365 * min_years_for_yearly_seasonality):
        forecast_comp, fitted_comp = _calculate_seasonal_median(train_dayofyear - 1, pred_dayofyear - 1, 366) # Dayofyear 1-366, map to 0-365
        secondary_seasonal_forecast_component += forecast_comp
        fitted_secondary_seasonal_on_train += fitted_comp
    return secondary_seasonal_forecast_component, fitted_secondary_seasonal_on_train
def _fit_predict_residual_correction_component(
    historical_data: np.ndarray, prediction_length: int, train_len: int, component_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates and applies a residual correction component.
    `historical_data` is expected to be finite.
    """
    rc_forecast_component = np.zeros(prediction_length, dtype=float)
    fitted_rc_on_train = np.zeros(train_len, dtype=float)
    if train_len == 0:
        return rc_forecast_component, fitted_rc_on_train
    default_rc_fit_window = max(5, prediction_length)
    rc_fit_window = max(1, min(component_config.get('rc_window_size', default_rc_fit_window), train_len))
    if rc_fit_window == 0:
        return rc_forecast_component, fitted_rc_on_train
    residual_correction_damping_factor = component_config.get('residual_correction_damping_factor', 1.0)
    
    residuals_to_correct = historical_data[-rc_fit_window:] # This data is finite.
    correction_method = component_config.get('residual_correction_method', 'mean')
    residual_correction_value = 0.0
    if correction_method == 'median':
        residual_correction_value = np.median(residuals_to_correct)
    else: # Default to 'mean'
        residual_correction_value = np.mean(residuals_to_correct)
    
    # Fitted: The amount of correction that would have been applied to the training window.
    # For simplicity and consistency with the "average residual" idea, apply the constant
    # correction value to the window from which it was derived. This represents what was "removed"
    # from these residuals.
    fitted_rc_on_train[-rc_fit_window:] = residual_correction_value
    # Forecast: Apply damping factor for forecast, decaying with prediction horizon.
    residual_correction_decay_enabled = component_config.get('residual_correction_decay_enabled', True)
    if residual_correction_decay_enabled:
        rc_forecast_component = residual_correction_value * (residual_correction_damping_factor ** np.arange(prediction_length))
    else:
        # If no decay, apply constant value without any damping factor
        rc_forecast_component = np.full(prediction_length, residual_correction_value)
    return rc_forecast_component, fitted_rc_on_train
# Main forecasting function
def fit_and_predict_fn(input_targets: pd.Series, prediction_index: pd.Index, season_length: int, config: Dict[str, Any]) -> pd.Series:
    """
    Forecasting function implementing an additive model with configurable components,
    now with an optional log transformation for multiplicative modeling.
    Components are applied sequentially, with each subsequent component learning on the residuals
    from the previous ones, akin to a gradient boosting approach.
    It robustly handles NaNs and provides ultimate fallbacks for edge cases.
    """
    prediction_length = len(prediction_index)
    train_len = len(input_targets)
    transform_log = config.get('transform_log', False)
    non_negative_constraint = config.get('non_negative', False)
    # --- 1. Robust NaN Handling for input_targets and determining initial base_level for processing ---
    original_targets_np = input_targets.values
    # Determine a robust fallback value from the original, finite data points.
    finite_original_values = original_targets_np[np.isfinite(original_targets_np)]
    if len(finite_original_values) > 0:
        initial_base_level_fallback_val = np.nanmedian(finite_original_values)
    else:
        initial_base_level_fallback_val = 0.0
    # Apply ffill, bfill using a temporary Pandas Series for convenience.
    temp_series = pd.Series(original_targets_np, index=input_targets.index)
    filled_targets_np = temp_series.ffill().bfill().values
    # Ensure all values in filled_targets_np are finite, using initial_base_level_fallback_val
    processed_targets_np = np.nan_to_num(filled_targets_np, nan=initial_base_level_fallback_val, posinf=initial_base_level_fallback_val, neginf=initial_base_level_fallback_val)
    # --- 2. Handle Edge Case: Empty or Very Short Processed Input ---
    if train_len == 0:
        # Determine the fallback value on the processed scale
        fallback_val_on_processed_scale = initial_base_level_fallback_val
        if transform_log:
            # If log-transform and non-negative, the lowest log-value for a non-negative original is log1p(0) = 0.
            if non_negative_constraint:
                fallback_val_on_processed_scale = 0.0
            else:
                # If not non-negative, can use log1p directly, but ensure initial_base_level_fallback_val is not too small
                fallback_val_on_processed_scale = np.log1p(np.maximum(0, initial_base_level_fallback_val))
        # Create predictions on the processed scale
        predictions_on_processed_scale = np.full(prediction_length, fallback_val_on_processed_scale)
        
        # Transform back to original scale if log transformation was applied
        if transform_log:
            predictions = np.expm1(predictions_on_processed_scale)
        else:
            predictions = predictions_on_processed_scale
        # Apply non-negativity constraint
        if non_negative_constraint:
            predictions = np.maximum(0, predictions)
            
        return pd.Series(predictions, index=prediction_index, name=f"{input_targets.name}_predictions")
    # Apply log transformation if configured, ensuring values are non-negative for log1p
    if transform_log:
        processed_targets_np = np.log1p(np.maximum(0, processed_targets_np))
    # --- 3. Prepare Seasonal Naive Fallback (always available for robustness) ---
    effective_season_length_for_naive = max(1, season_length)
    base_seasonal_pattern_naive_processed_scale = processed_targets_np[-effective_season_length_for_naive:]
    if len(base_seasonal_pattern_naive_processed_scale) == 0:
        # Re-evaluate fallback on processed (potentially log) scale for this path.
        if transform_log:
            if non_negative_constraint:
                fallback_val_for_naive = 0.0
            else:
                fallback_val_for_naive = np.log1p(np.maximum(0, initial_base_level_fallback_val))
        else:
            fallback_val_for_naive = initial_base_level_fallback_val
        base_seasonal_pattern_naive_processed_scale = np.array([fallback_val_for_naive])
    effective_pattern_length_naive = max(1, len(base_seasonal_pattern_naive_processed_scale))
    num_repeats_naive = (prediction_length + effective_pattern_length_naive - 1) // effective_pattern_length_naive
    seasonal_naive_fallback_predictions_processed_scale = np.tile(base_seasonal_pattern_naive_processed_scale, num_repeats_naive)[:prediction_length]
    
    if transform_log:
        seasonal_naive_fallback_predictions = np.expm1(seasonal_naive_fallback_predictions_processed_scale)
    else:
        seasonal_naive_fallback_predictions = seasonal_naive_fallback_predictions_processed_scale
    seasonal_naive_fallback_predictions = np.maximum(0, seasonal_naive_fallback_predictions)
    # --- 4. Initialize Additive Model Components ---
    component_functions = {
        'base_level': _fit_predict_base_level,
        'trend': _fit_predict_trend_component,
        'seasonal_primary': _fit_predict_seasonal_primary_component,
        'seasonal_secondary': _fit_predict_seasonal_secondary_component,
        'residual_correction': _fit_predict_residual_correction_component,
    }
    base_level_config = next((c for c in config.get('components', []) if c.get('type') == 'base_level'), None)
    if base_level_config is None:
        base_level_config = {'type': 'base_level', 'base_level_method': 'median_all_history'}
    # processed_targets_np is guaranteed finite.
    initial_base_forecast, initial_base_fitted_on_train = component_functions['base_level'](
        historical_data=processed_targets_np,
        prediction_length=prediction_length,
        train_len=train_len,
 
        season_length=season_length, # Pass original season_length to base_level
        component_config=base_level_config
    )
    predictions_on_processed_scale = initial_base_forecast
    # current_residuals will be finite as processed_targets_np and initial_base_fitted_on_train are finite.
    current_residuals = processed_targets_np - initial_base_fitted_on_train
    # --- 5. Sequentially apply other components (boosting-like approach) ---
    for component_config in config.get('components', []):
        component_type = component_config.get('type')
 
        if component_type == 'base_level': # Base level already handled
            continue
        # Determine the effective season length for this specific component
        effective_comp_season_length = season_length # Start with the provided season_length from GluonTS
        # Special handling for seasonal_primary to infer a more relevant season_length if GluonTS defaults to 1
        if component_type == 'seasonal_primary' and effective_comp_season_length <= 1:
            freq_str = input_targets.index.freqstr if input_targets.index.freqstr is not None else ''
            
            # Calculate data span in years for yearly seasonality checks, consistent with secondary seasonality
            data_span_days = (input_targets.index.max() - input_targets.index.min()).days if train_len > 1 else 0
            min_years_for_yearly_seasonality = 2.0
            
            # Prioritize weekly seasonality for daily data if sufficient history
            if 'D' in freq_str and train_len >= 2 * 7: # At least two full weeks of data
                effective_comp_season_length = 7
            # Prioritize yearly seasonality for monthly data if sufficient history
            elif 'M' in freq_str and data_span_days >= (365 * min_years_for_yearly_seasonality):
                effective_comp_season_length = 12
            # Prioritize yearly seasonality for weekly data if sufficient history
            elif 'W' in freq_str and data_span_days >= (365 * min_years_for_yearly_seasonality):
                effective_comp_season_length = 52
            # Add yearly seasonality for daily if not already picked up by weekly and sufficient history
            elif 'D' in freq_str and data_span_days >= (365 * min_years_for_yearly_seasonality):
                effective_comp_season_length = 365
            
            # If after all checks, effective_comp_season_length is still <= 1,
            # then no meaningful primary season was inferred, and the component will return zeros.
        if component_type in component_functions:
 
            func_args = {
                'historical_data': current_residuals, # This input is guaranteed finite
                'prediction_length': prediction_length,
                'train_len': train_len,
                'component_config': component_config,
            }
 
 
 
            if component_type == 'seasonal_primary': # Pass the derived value
                func_args['season_length'] = effective_comp_season_length
            elif component_type == 'seasonal_secondary':
                func_args['input_targets_index'] = input_targets.index
                func_args['prediction_index'] = prediction_index
            elif component_type == 'trend': # Trend uses original season_length for windowing
                func_args['season_length'] = season_length
            # Base_level and residual_correction do not need `season_length` from here,
            # as base_level already received it during initialization and residual_correction doesn't use it.
            
            component_forecast, component_fitted_on_train = component_functions[component_type](**func_args)
            predictions_on_processed_scale += component_forecast
            current_residuals -= component_fitted_on_train
    # --- 6. Transform back if log transformation was applied ---
    if transform_log:
        # Ensure non-negativity on the log-transformed scale if the original output should be non-negative.
        # np.expm1(x) is >= 0 if and only if x >= 0.
        if non_negative_constraint:
            predictions_on_processed_scale = np.maximum(0, predictions_on_processed_scale)
        predictions = np.expm1(predictions_on_processed_scale)
    else:
        predictions = predictions_on_processed_scale
    # --- 7. Final Robustness Checks ---
    # Fall back to seasonal naive if predictions contain any NaNs or Infs
    if not np.all(np.isfinite(predictions)):
        predictions = seasonal_naive_fallback_predictions
    # --- 8. Apply Non-Negativity Constraint if configured ---
    # This acts as a final safeguard, even if non_negative_constraint was applied on log scale.
    if non_negative_constraint:
        predictions = np.maximum(0, predictions)
    return pd.Series(predictions, index=prediction_index, name=f"{input_targets.name}_predictions")