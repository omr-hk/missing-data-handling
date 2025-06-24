import warnings
warnings.filterwarnings(action = 'ignore')
import numpy as np
import pandas as pd
import math
import os
import glob
from datetime import datetime
from sklearn.metrics import root_mean_squared_error
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler
import random
pd.set_option("display.precision", 10)
from tqdm import tqdm
import concurrent.futures
import statsmodels.api as sm

EARTH_RADIUS_KM = 6378.137
MU = 398600

def calculate_orbital_elements(mean_motion, eccentricity):
    mean_motion_rad_per_s = mean_motion * (2 * math.pi) / 86400
    a = (MU / mean_motion_rad_per_s**2) ** (1/3)
    apogee = a * (1 + eccentricity)
    perigee = a * (1 - eccentricity)
    
    return a, apogee, perigee

def calculate_orbital_energy(semi_major_axis): 
    return ( - MU / (2*semi_major_axis) )

def calculate_altitude(semi_major_axis):
    return semi_major_axis - EARTH_RADIUS_KM


def prep_file(path):
    threshold = pd.Timedelta(days=2)
    df_test = pd.read_csv(path)
    df_test['EPOCH_DATE'] = pd.to_datetime(df_test['EPOCH_DATE'], format= '%d-%m-%y')
    df_test['EPOCH_TIME'] = pd.to_datetime(df_test['EPOCH_TIME'],format= '%H:%M:%S:%f').dt.time
    df_test['EPOCH_DATE_TIME'] = df_test.apply(lambda row: datetime.combine(row['EPOCH_DATE'].date(), row['EPOCH_TIME']), axis=1)
    df_test['SEMI_MAJOR_AXIS'], df_test['APOGEE'], df_test['PERIGEE'] = calculate_orbital_elements(df_test['MEAN_MOTION'],df_test['ECCENTRICITY'])
    df_test['ORBITAL_ENERGY'] = calculate_orbital_energy(df_test['SEMI_MAJOR_AXIS'])
    df_test['ALTITUDE'] = calculate_altitude(df_test['SEMI_MAJOR_AXIS'])
    #Group rows having same date and time by taking maximum apogee or minimum perigee
    df_test = df_test.loc[df_test.groupby(['EPOCH_DATE', 'EPOCH_TIME'])['APOGEE'].idxmax()]
    df_test = df_test.sort_values('EPOCH_DATE_TIME')
    df_test = df_test.drop_duplicates()
    df_test.set_index(df_test['EPOCH_DATE_TIME'], inplace=True)
    df_test.sort_index(inplace=True)
    return df_test

def find_valid_n_day_windows(data, column, n, alpha=0.05):

    # Rolling windows of n days
    rolling_windows = data[column].rolling(window=n)

    # Keep only fully populated windows (no NaN values)
    valid_windows = rolling_windows.count() == n
    valid_indices = valid_windows[valid_windows].index.to_list()[1:-n]
    
    # Label each window trend usin Ordinary Linear Model slope 
    trend_labels = []
    window_indices = []

    for idx in valid_indices:
        window = data.loc[idx:idx + pd.Timedelta(days=n - 1)]
        if window[column].hasnans:
            continue
        
        x = np.arange(n)
        X = sm.add_constant(x) 

        model = sm.OLS(window[column], X).fit()
        slope = model.params[1]
        pvalue = model.pvalues[1]

        if pvalue < alpha:
            if slope > 0:
                trend = 'increasing'
            elif slope < 0:
                trend = 'decreasing'
        else:
            trend = 'stable'

        
        trend_labels.append(trend)
        window_indices.append(window.index.to_list())
        
    selected_windows = {}
    trend_types = ['increasing', 'decreasing', 'stable']

    for trend in trend_types:
        matching_windows = [window for i, window in enumerate(window_indices) if trend_labels[i] == trend]

        if matching_windows:
            selected_windows[trend] = random.choice(matching_windows)
    
    return selected_windows

def compute_error_variance_metric(l, lamb):
    return np.mean(l) + (lamb * np.std(l))

def get_interpolation_metrics(df):
    interpolation_methods = [
        'polynomial', 'spline','linear', 'time', 'index', 'values', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline', 'from_derivatives'
    ]
    trends = ['increasing', 'decreasing', 'stable']
    ndays = [2, 3, 5, 7]
    cols = ['B*', 'ECCENTRICITY', 'MEAN_MOTION', 'APOGEE']
    polynomial_orders = [2, 3, 4, 5]
    scaler = MinMaxScaler()
    test_r = df[cols].copy()
    test_r['date'] = test_r.index.date
    #Grouping same dates by max apogee or min perigee
    idx = test_r.groupby('date')['APOGEE'].idxmax()
    test = test_r.loc[idx].copy()
    test.index = test.index.date
    full_date_range = pd.date_range(start=test.index.min(), end=test.index.max(), freq="D")
    test = test.reindex(full_date_range)
    test.sort_index(inplace=True)
    test.drop(['date', 'APOGEE'], axis=1, inplace=True)
    cols.remove('APOGEE')
    test[cols] = scaler.fit_transform(test)
    methods = []
    orders = []
    rmse_list = []
    lam = 1.0
    for mid, method in enumerate(interpolation_methods):
        if mid < 2:
            for order in polynomial_orders:
                ndays_rmse_list = []
                for n in ndays:
                    cols_rmse_list = []
                    for col in cols:
                        trend_rmse_list = []
                        trend_batches = find_valid_n_day_windows(test.copy(),col, n)
                        for trend in trends:
                            if trend in trend_batches:
                                try:
                                    current_batch = trend_batches[trend]
                                    temp = test.copy()
                                    true_values = temp.loc[current_batch, col].copy()
                                    temp.loc[current_batch, col] = np.nan
                                    interpolated = temp[col].astype(float).interpolate(method=method, order=order)
                                    batch_rmse = root_mean_squared_error(true_values, interpolated.loc[true_values.index])
                                    trend_rmse_list.append(batch_rmse)
                                except Exception as e:
                                    print("ERROR")
                                    print(f"METHOD: {method} ORDER: {order}")
                                    print(e)
                        col_evs = compute_error_variance_metric(trend_rmse_list, lam)
                        cols_rmse_list.append(col_evs)
                    ndays_evs = compute_error_variance_metric(cols_rmse_list, lam)
                    ndays_rmse_list.append(ndays_evs)
                order_evs = compute_error_variance_metric(ndays_rmse_list, lam)
                rmse_list.append(order_evs)
                methods.append(method)
                orders.append(order)
        
        else:
            ndays_rmse_list = []
            for n in ndays:
                cols_rmse_list = []
                for col in cols:
                    trend_rmse_list = []
                    trend_batches = find_valid_n_day_windows(test.copy(),col, n)
                    for trend in trends:
                        if trend in trend_batches:
                            try:
                                current_batch = trend_batches[trend]
                                temp = test.copy()
                                true_values = test.loc[current_batch, col]
                                temp.loc[current_batch, col] = np.nan
                                interpolated = temp[col].astype(float).interpolate(method=method)
                                batch_rmse = root_mean_squared_error(true_values, interpolated.loc[true_values.index])
                                trend_rmse_list.append(batch_rmse)
                            except Exception as e:
                                print("ERROR")
                                print(f"METHOD: {method}")
                                print(e)
                    col_evs = compute_error_variance_metric(trend_rmse_list, lam)
                    cols_rmse_list.append(col_evs)
                ndays_evs = compute_error_variance_metric(cols_rmse_list, lam)
                ndays_rmse_list.append(ndays_evs)
            method_evs = compute_error_variance_metric(ndays_rmse_list, lam)
            rmse_list.append(method_evs)
            methods.append(method)
            orders.append(0)
    clear_output(wait=True)
    return pd.DataFrame(
        {
            'method' : methods,
            'order' : orders,
            'evm' : rmse_list
        }
    )


def process_single_directory(directory):
    try:
        # read files
        data_set_path = f'{directory}/*.csv'
        dataset = glob.glob(data_set_path)
        if not dataset:
            return []  # No CSV found in this directory
        # Process the first CSV file found
        df = prep_file(dataset[0])
        result = get_interpolation_metrics(df)
        # Compute ranks (lower EVM means better performance)
        result['rank'] = result['evm'].rank(method='dense', ascending=True).fillna(0).astype(int)
        file_results = []
        for _, row in result.iterrows():
            file_results.append({
                'method': row['method'],
                'order': row['order'],
                'rank': f"Rank {int(row['rank'])}",
                'evm': row['evm']  
            })
        return file_results
    except Exception as e:
        # Log error and return an empty list
        print(f"Error processing directory {directory}: {e}")
        return []


def aggregate_results(results_list):
    # Counting ranks for each methods
    final_df = pd.DataFrame(columns=['method', 'order'])
    
    for file_results in results_list:
        for res in file_results:
            method = res['method']
            order = res['order']
            rank_label = res['rank']
            method_filter = (final_df['method'] == method) & (final_df['order'] == order)
            if method_filter.any():
                row_idx = final_df.index[method_filter][0]
                if rank_label in final_df.columns:
                    final_df.at[row_idx, rank_label] += 1
                else:
                    final_df.loc[row_idx, rank_label] = 1
            else:
                new_row = {'method': method, 'order': order, rank_label: 1}
                final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
                final_df.fillna(0, inplace=True)
    return final_df


if __name__ == '__main__':
    SAMPLE_ROOT_PATH = './Data/datastore'
    files = [x[0] for x in os.walk(SAMPLE_ROOT_PATH)][1:]
    all_results = []
    of_name = "apogee"
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_directory, file): file for file in files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing directories"):
            try:
                file_result = future.result()
                all_results.append(file_result)
            except Exception as e:
                print("Exception encountered:", e)

    #aggregate results from all files
    final_df = aggregate_results(all_results)
    final_df.to_excel(f'{of_name}.xlsx', index=False)
    print(f"Processing complete. Results saved to {of_name}.xlsx")

    file_path = f"{of_name}.xlsx"  # file path
    df = pd.read_excel(file_path)

    # Identify columns
    fixed_columns = ['method', 'order']  # Fixed columns that should stay in place
    rank_columns = [col for col in df.columns if col.startswith("Rank")]

    sorted_rank_columns = sorted(rank_columns, key=lambda x: int(x.split()[1]))
    sorted_rank_columns = sorted_rank_columns[1:]

    df = df[fixed_columns + sorted_rank_columns]

    # Compute total count for each Method-Order pair
    df['Total'] = df[sorted_rank_columns].sum(axis=1)

    # Compute proportions
    df_proportion = df.copy()
    df_proportion[sorted_rank_columns] = df[sorted_rank_columns].div(df['Total'], axis=0)*100

    # Drop the Total column in proportion sheet
    df_proportion.drop(columns=['Total'], inplace=True)
    df.drop(columns=['Total'], inplace=True)
    # Save both sheets in a new Excel file
    output_file = f"{of_name}_final.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="Raw Counts", index=False)
        df_proportion.to_excel(writer, sheet_name="Proportions", index=False)

    print(f"File saved as {output_file}")