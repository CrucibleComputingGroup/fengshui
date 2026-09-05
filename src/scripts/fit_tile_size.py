import numpy as np
from scipy import optimize
from scipy import stats

def fit_power_function(x_values, y_values, outlier_threshold=2.0, min_points=3):
    """
    Fits a power function of the form y = k * x^b + t to the given data points.
    Also tries exponential and logarithmic functions and returns the best fit.
    
    Args:
        x_values (list or array): Independent variable values
        y_values (list or array): Dependent variable values
        outlier_threshold (float): Z-score threshold for outlier removal (default: 2.0)
        min_points (int): Minimum number of points required for fitting (default: 4)
    
    Returns:
        tuple: (params, r_squared) where:
            params (dict): Contains parameters of the best fitting function
            r_squared (float): R² value of the best fit
    """
    # Convert inputs to numpy arrays
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    # Step 1: Remove all (x,y) when y is 0/inf
    # Create a single mask for all conditions
    valid_mask = (y != 0) & np.isfinite(y)
    
    # Apply mask to keep only valid data points
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Check if we have enough data after removing zeros
    if len(x) < min_points:
        # Not enough points for a proper fit, return None type with the original data
        x_filtered_str = ','.join([str(val) for val in x])
        y_filtered_str = ','.join([str(val) for val in y])
        
        return {
            'type': 'None',
            'r_squared': 0.0,
            'k': 0.0,
            'b': 0.0,
            't': 0.0,
            'x_filtered': x_filtered_str,
            'y_filtered': y_filtered_str
        }, 0.0
    
    # Step 2: Enforce monotonicity starting from smallest x
    idx_sorted = np.argsort(x)
    x_sorted = x[idx_sorted]
    y_sorted = y[idx_sorted]
    
    x_monotonic = [x_sorted[0]]
    y_monotonic = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        curr_x = x_sorted[i]
        curr_y = y_sorted[i]
        
        # Check if adding this point maintains monotonicity
        if curr_y >= y_monotonic[-1]:
            # Point maintains monotonicity, add it
            x_monotonic.append(curr_x)
            y_monotonic.append(curr_y)
        else:
            # The point breaks monotonicity
            
            # First check: removing the previous point
            if len(y_monotonic) > 1 and y_monotonic[-2] <= curr_y:
                # We can remove just the last point and add the current one
                remaining_after_removal = len(y_monotonic) - 1 + 1 + (len(x_sorted) - i - 1)
                
                if remaining_after_removal >= min_points:
                    # Safe to remove the last point
                    x_monotonic.pop()
                    y_monotonic.pop()
                    x_monotonic.append(curr_x)
                    y_monotonic.append(curr_y)
            else:
                # Need to check for any prior point that's larger than current
                points_to_keep = []
                
                # Find all points that should be kept
                for j in range(len(y_monotonic)):
                    if y_monotonic[j] <= curr_y:
                        points_to_keep.append(j)
                
                # Will we have enough points if we remove all violators?
                remaining_points = len(points_to_keep) + 1 + (len(x_sorted) - i - 1)
                
                if remaining_points >= min_points:
                    # Safe to remove violating points
                    x_new = [x_monotonic[j] for j in points_to_keep]
                    y_new = [y_monotonic[j] for j in points_to_keep]
                    
                    x_monotonic = x_new
                    y_monotonic = y_new
                    x_monotonic.append(curr_x)
                    y_monotonic.append(curr_y)
        
        # Check if we have enough points remaining
        remaining_points = len(x_monotonic) + (len(x_sorted) - i - 1)
        if remaining_points < min_points:
            # We won't have enough points even if we add all remaining ones
            break
    
    # Convert lists back to numpy arrays
    x_monotonic = np.array(x_monotonic)
    y_monotonic = np.array(y_monotonic)
    
    # Step 3: Remove outliers based on z-score if enough points remain
    if len(x_monotonic) > min_points:
        # Calculate z-scores for y values
        y_z_scores = np.abs(stats.zscore(y_monotonic))
        
        # Check if removing outliers would leave us with enough points
        outliers_count = np.sum(y_z_scores > outlier_threshold)
        if len(x_monotonic) - outliers_count >= min_points:
            inlier_mask = y_z_scores <= outlier_threshold
            x_monotonic = x_monotonic[inlier_mask]
            y_monotonic = y_monotonic[inlier_mask]
    
    # If we don't have enough points after filtering, use the original data
    if len(x_monotonic) < min_points:
        x_monotonic = x
        y_monotonic = y
    
    # Step 4: Try different function types
    
    # 1. Power function: y = k * x^b + t
    def power_func(x, k, b, t):
        # Prevent overflow by using log-domain computation for large values
        try:
            # Limit the exponent to avoid overflow
            safe_b = np.clip(b, -100, 100)
            return k * np.power(x, safe_b) + t
        except:
            # Fallback for any computation issues
            return np.ones_like(x) * (np.mean(y_monotonic))
    
    # 3. Logarithmic function: y = a * log(b * x) + c
    def log_func(x, a, b, c):
        # Ensure x*b is positive for log
        return a * np.log(np.maximum(b * x, 1e-10)) + c
    
    # Determine y range for initial guesses
    y_max = np.max(y_monotonic)
    y_min = np.min(y_monotonic)
    
    # Results storage
    results = []
    
    # 1. Try fitting power function
    try:
        # Adjust initial guess based on y range
        if y_max < 1e-3:  # Small y values
            k_init = 1e-5
            b_init = 1.0
            t_init = 0.0
        elif y_min > 1000:  # Large y values
            k_init = 1000
            b_init = 1.0
            t_init = 0.0
        else:  # Default range
            k_init = 1.0
            b_init = 1.0
            t_init = 0.0
        
        # Try using robust fitting methods with more constrained bounds
        try:
            popt, _ = optimize.curve_fit(
                power_func, x_monotonic, y_monotonic,
                p0=[k_init, b_init, t_init],
                bounds=([0, -10, -np.inf], [np.inf, 10, np.inf]),  # Constrain b to avoid overflow
                maxfev=5000,
                method='trf'  # Use trust region reflective algorithm which is more robust
            )
            
            k_fit, b_fit, t_fit = popt
            y_pred = power_func(x_monotonic, k_fit, b_fit, t_fit)
            residuals = y_monotonic - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_monotonic - np.mean(y_monotonic)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Only accept the fit if it's reasonable
            if np.isfinite(r_squared) and not np.isnan(r_squared) and r_squared > -0.1:
                results.append(({
                    'type': 'power',
                    'k': k_fit,
                    'b': b_fit,
                    't': t_fit
                }, r_squared))
            
        except Exception as e:
            # Silently continue if this fit fails
            pass
        
    except Exception as e:
        pass
    
    # 3. Try fitting logarithmic function
    try:
        # Initial guesses
        if y_max < 1e-3:
            a_init = 1e-5
        elif y_min > 1000:
            a_init = 1000
        else:
            a_init = 1.0
        
        popt, _ = optimize.curve_fit(
            log_func, x_monotonic, y_monotonic,
            p0=[a_init, 1.0, 0.0],
            maxfev=10000
        )
        
        a_fit, b_fit, c_fit = popt
        y_pred = log_func(x_monotonic, a_fit, b_fit, c_fit)
        residuals = y_monotonic - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_monotonic - np.mean(y_monotonic)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results.append(({
            'type': 'logarithmic',
            'a': a_fit,
            'b': b_fit,
            'c': c_fit
        }, r_squared))
        
    except Exception as e:
        pass

    # add value at 1 add the result
    if 1 not in x_monotonic:
        # Find the corresponding y value from x_sorted (regular list) and y_sorted
        idx = np.where(x_sorted == 1)[0][0]
        y_value = y_sorted[idx]  # Get the corresponding y value
        
        # Prepend 1 and its y value to the numpy arrays
        x_monotonic = np.insert(x_monotonic, 0, 1)
        y_monotonic = np.insert(y_monotonic, 0, y_value)
    
    # If no fits succeeded or all gave poor results, use None type
    if not results or max(result[1] for result in results) < 0.1:
        # Convert filtered data points to strings
        x_filtered_str = ','.join([str(x) for x in x_monotonic])
        y_filtered_str = ','.join([str(y) for y in y_monotonic])
        
        # Return None type with just the filtered data
        return {
            'type': 'None',
            'r_squared': 0.0,
            'k': 0.0,
            'b': 0.0,
            't': 0.0,
            'x_filtered': x_filtered_str,
            'y_filtered': y_filtered_str
        }, 0.0
    
    # Find the best fit
    if results:
        best_result = max(results, key=lambda x: x[1])
        best_params, best_r2 = best_result
        
        # Create a standardized parameter structure
        function_type = best_params['type']
        
        # Convert filtered data points to strings
        x_filtered_str = ','.join([str(x) for x in x_monotonic])
        y_filtered_str = ','.join([str(y) for y in y_monotonic])
        
        # Base structure that all function types will have
        standard_params = {
            'type': function_type,
            'r_squared': best_r2,
            'x_filtered': x_filtered_str,
            'y_filtered': y_filtered_str
        }
        
        # Add specific parameters for each function type
        if function_type == 'power':
            standard_params.update({
                'k': float(best_params['k']),
                'b': float(best_params['b']),
                't': float(best_params['t'])
            })
        elif function_type == 'logarithmic':
            # Map logarithmic parameters to k/b/t
            standard_params.update({
                'k': float(best_params['a']),
                'b': float(best_params['b']),
                't': float(best_params['c'])
            })
        elif function_type == 'None':
            # Leave k, b, t as default values
            standard_params.update({
                'k': 0.0,
                'b': 0.0,
                't': 0.0
            })
        
        best_params = standard_params
    
    # Print debugging information if fit is poor
    if best_r2 < 0.7:
        print(x)
        print(y)
        print(x_monotonic)
        print(y_monotonic)
        
        if function_type == 'power':
            print(f"Fitted function (power): y = {best_params['k']:.6f} * x^{best_params['b']:.6f} + {best_params['t']:.6f}")
        elif function_type == 'logarithmic':
            print(f"Fitted function (logarithmic): y = {best_params['k']:.6f} * log({best_params['b']:.6f} * x) + {best_params['t']:.6f}")
        elif function_type == 'None':
            print("No suitable function fit found")
        
        print(f"R-squared: {best_r2:.6f}")
        print(f"Number of points: Original={len(x)}, After filtering={len(x_monotonic)}")
    
    return best_params, best_r2