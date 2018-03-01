# %%

import psycopg2
import config
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from scipy import stats
import GPy

plt.close('all')


def connect_db(dbname):
    if dbname != config.db_dict['dbname']:
        raise ValueError("Couldn't not find DB with given name")
    conn = psycopg2.connect(
        host=config.db_dict['host'],
        user=config.db_dict['user'],
        password=config.db_dict['password'],
        dbname=config.db_dict['dbname'])
    return conn


query = """
SELECT
  result_time,
  node_id,
  0.00119 * adc0 - 0.401 AS sm_30cm,
  0.00119 * adc1 - 0.401 AS sm_10cm
FROM node_sensor_data
WHERE adc0 BETWEEN 350 AND 1100
      AND adc1 BETWEEN 350 AND 1100
      AND voltage > 3000
      AND (node_id % 100 = 51 OR node_id % 100 = 31)
      AND result_time BETWEEN '2016-07-01' AND '2016-09-01'
ORDER BY result_time;
"""

old_df = pd.read_sql_query(
    query, connect_db('audubon_prev'), index_col='result_time')
old_df['node_id'] = old_df['node_id'].astype(int)

print(old_df.head(), old_df.tail(), sep='\n')


def get_consec_times(series, gap_time, consec_time):
    series = series.dropna()
    start_diffs = series.index.to_series().diff()
    end_diffs = series.index.to_series().diff(-1)
    starts = list(start_diffs[start_diffs > pd.Timedelta(gap_time)].index)
    ends = list(end_diffs[end_diffs < -pd.Timedelta(gap_time)].index)
    starts.append(min(series.index))
    ends.append(max(series.index))
    start_end_tups = zip(sorted(starts), sorted(ends))
    start_end_tups = [
        tup for tup in start_end_tups
        if (tup[1] - tup[0] > pd.Timedelta(consec_time))
    ]
    return start_end_tups


def squash_spikes(ser_in):
    ser_out = ser_in.copy()
    ser_d2 = ser_in.rolling(
        30, center=True,
        min_periods=1).apply(lambda x: np.mean(np.diff(a=x, n=2)))
    local_std = ser_in.rolling(20, min_periods=1, center=True).std()
    # locate spikes
    spike_idx = ser_in.index[(ser_d2 <= 0) & (local_std > ser_in.std() * 0.8)]
    # exit if no spikes are found
    if len(spike_idx) == 0:
        return ser_in, []
    # create buffer around spikes
    buffer_idx = []
    for idx in spike_idx:
        start, end = idx - pd.Timedelta('3h'), idx + pd.Timedelta('6h')
        buffer_idx.append(pd.Series(ser_in[start:end].index))
    buffer_idx = pd.DatetimeIndex(pd.concat(buffer_idx).drop_duplicates())
    # loop for each separate interval around spike
    start_end_tups = get_consec_times(ser_in[buffer_idx], '1h', '6h')
    cut_quant = 0.5
    spike_start_end_tups = []
    for (start, end) in start_end_tups:
        consec_idx = ser_in[buffer_idx].loc[start:end][:-1].dropna().index
        consec = ser_in[consec_idx].copy()
        spike_idx = consec[consec > consec.quantile(cut_quant)].index

        spike_min = ser_in[spike_idx].min()
        spike_max = ser_in[spike_idx].max()
        approach_val = ser_in[consec_idx].quantile(cut_quant)
        rate = 1

        ser_out[spike_idx] = (ser_out[spike_idx] - spike_min) / (
                spike_max - spike_min)
        ser_out[spike_idx] = ser_out[spike_idx] / (
                ser_out[spike_idx] / approach_val + 1 / rate)
        ser_out[spike_idx] = ser_out[spike_idx] * (
                spike_max - spike_min) + spike_min

        spike_start_end_tups.append((min(spike_idx) - pd.Timedelta('40min'),
                                     max(spike_idx) + pd.Timedelta('5min')))
    return ser_out, spike_start_end_tups


def remove_batt_tails(ser):
    ser = ser.sort_index().copy()
    bad_start = max(ser.index) + pd.Timedelta('1h')
    bad_end = bad_start
    ser = ser.rolling(30, center=True, min_periods=1).mean()
    tail_len = 8
    for i in range(1, 100):
        tail = ser.sort_index().iloc[(-tail_len - i):(-i)]
        x = np.arange(0, tail_len)
        y = tail.values
        if len(x) != len(y):
            continue
        slope_tail, _, _, _, _ = stats.linregress(x, y)

        before_tail = ser.sort_index().iloc[(-2 * tail_len - i):(
                -tail_len - i)]
        x = np.arange(0, tail_len)
        y = before_tail.values
        if len(x) != len(y):
            continue
        slope_before, _, _, _, _ = stats.linregress(x, y)
        if (before_tail.median() < tail.median()) \
            | (slope_before + 0.03 < slope_tail) \
            | (tail.median() > ser.quantile(0.3)) \
            | (slope_tail > -0.00028):
            break
    if i > 1:
        bad_start = ser.index[-(tail_len + i)] - pd.Timedelta('10min')
        ser = ser.iloc[:-(tail_len + i)]
    return ser, bad_start, bad_end


old_df_chopped = old_df.copy()
for count, (node_id, node_group) in enumerate(old_df.groupby('node_id')):
    if count != 16:
        old_df_chopped.drop(
            old_df_chopped[old_df_chopped['node_id'] == node_id].index,
            inplace=True)
        continue
    print('--- {}'.format(node_id))
    start_end_tups = get_consec_times(node_group, '12h', '4h')
    for (start, end) in start_end_tups:
        consec = node_group.loc[start:end][:-1]
        for depth in ['sm_10cm', 'sm_30cm']:
            consec_sensor = consec[depth].rolling(
                15, center=True, min_periods=1).mean()
            data = consec_sensor.sort_index().reset_index()
            if len(data) < 10:
                continue
            # print(len(data))
            freq = '15T'
            xn = pd.date_range(
                min(data['result_time']).ceil(freq),
                max(data['result_time']).floor(freq),
                freq=freq)
            xn = xn.view('int64') // pd.Timedelta(1, unit='s')
            data['result_time'] = data['result_time'].view(
                'int64') // pd.Timedelta(
                1, unit='s')
            data = data.as_matrix()

            f = interpolate.BarycentricInterpolator(data[:, 0], data[:, 1])
            yn = f(xn)
            interp_ser = pd.Series(
                yn, index=pd.to_datetime(xn * pd.Timedelta(1, unit='s')), name=(node_id, depth))

            no_batt_tail_ser, bad_start, bad_end = remove_batt_tails(
                interp_ser)
            if bad_end != bad_start:
                old_df_chopped.loc[(old_df_chopped['node_id'] == node_id) & (
                        old_df_chopped.index < bad_end) & (
                                           old_df_chopped.index > bad_start), depth] = np.nan
            no_spike_ser, spike_start_end_tups = squash_spikes(
                no_batt_tail_ser)
            if len(spike_start_end_tups) != 0:
                for (start, end) in spike_start_end_tups:
                    old_df_chopped.loc[(old_df_chopped['node_id'] == node_id) &
                                       (old_df_chopped.index < end) &
                                       (old_df_chopped.index > start),
                                       depth] = np.nan
                    temp_df = pd.DataFrame()
                    temp_df[depth] = no_spike_ser[(no_spike_ser.index < end)
                                                  &
                                                  (no_spike_ser.index > start)]
                    temp_df['node_id'] = node_id
                    temp_df.resample('60min')
                    old_df_chopped = old_df_chopped.append(temp_df)


gp_dict = {}

for count, (node_id, node_group) in enumerate(
        old_df_chopped.groupby('node_id')):
    for depth in ['sm_10cm', 'sm_30cm']:
        col = node_group[depth].dropna()
        start_end_tups = get_consec_times(col, '24h', '1h')
        for (start, end) in start_end_tups:
            col = col.loc[start:end][:-1].dropna()
            if len(col) < 5:
                continue
            col_name = (node_id, depth)
            print(col_name, len(col))
            X = (col.index.values.astype(int) // 10**9).reshape(-1, 1)
            X_norm = (X - np.mean(X)) / np.std(X)
            X_pred_time = pd.date_range(
                min(col.index).round('15T'),
                max(col.index).round('15T'),
                freq='15T')
            print(X_pred_time)
            X_pred = (X_pred_time.values.astype(int) // 10**9).reshape(-1, 1)
            X_pred = (X_pred - np.mean(X)) / np.std(X)
            y = col.values.reshape(-1, 1)
            y_norm = (y - np.mean(y)) / np.std(y)
            # print(X_norm, y_norm)
            kernel = GPy.kern.Matern32(
                input_dim=1, variance=0.75,
                lengthscale=0.1) + GPy.kern.White(input_dim=1)
            m = GPy.models.GPRegression(X_norm, y_norm, kernel)
            print('{} {}\ndate range: {} --- {}\ninterval: {}\nnum points = {}'.
                  format(node_id, depth, start, end, end - start, len(X_norm)))
            m.optimize(messages=True)
            y_mean_norm, y_cov_norm = m.predict_noiseless(X_pred)
            quant_lo, quant_hi = m.predict_quantiles(X_pred, quantiles=(40, 60))
            # y_mean_norm[np.where((quant_hi - quant_lo) > np.percentile((quant_hi - quant_lo), 75))] = np.nan
            quant_lo = quant_lo * np.std(y) + np.mean(y)
            quant_hi = quant_hi * np.std(y) + np.mean(y)
            y_mean = y_mean_norm * np.std(y) + np.mean(y)
            quants_and_mean = np.hstack([quant_lo, y_mean, quant_hi])
            gp_ser = pd.Series(y_mean.reshape(-1), index=X_pred_time)
            gp_sub_df = pd.DataFrame(
                quants_and_mean,
                columns=['quant_lo', 'mean', 'quant_hi'],
                index=X_pred_time)
            # print(gp_ser)
            if col_name not in gp_dict.keys():
                gp_dict[col_name] = [gp_sub_df]
            else:
                gp_dict[col_name].append(gp_sub_df)

gp_dict = {k: pd.concat(v) for k, v in gp_dict.items()}
gp_df = pd.concat(gp_dict, axis=1)
gp_df.columns.rename(['node_id', 'sm_depth', 'quantile'], inplace=True)



for name, df in gp_df.groupby(level=[0,1], axis=1):
    print(name)
    fig, ax = plt.subplots(figsize=(11, 7))
    for col_name, col in df.iteritems():
        ax.plot(col.index, col, c='w', linewidth=10, alpha=0.6)
        ax.plot(col.index, col, c='k', linewidth=5)
    old_ser = old_df.loc[old_df['node_id'] == col_name[0], col_name[1]]
    old_chopped_ser = old_df_chopped.loc[old_df_chopped['node_id'] == col_name[
        0], col_name[1]]
    ax.scatter(old_ser.index, old_ser, c='r')
    ax.scatter(old_chopped_ser.index, old_chopped_ser, c='b')
    plt.tight_layout()
    plt.show()


# for col_name, col in gp_df.iteritems():
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(col.index, col, c='w', linewidth=10, alpha=0.6)
#     ax.plot(col.index, col, c='k', linewidth=5)
#     old_ser = old_df.loc[old_df['node_id'] == col_name[0], col_name[1]]
#     old_chopped_ser = old_df_chopped.loc[old_df_chopped['node_id'] == col_name[
#         0], col_name[1]]
#     ax.scatter(old_ser.index, old_ser, c='r')
#     ax.scatter(old_chopped_ser.index, old_chopped_ser, c='b')
#     plt.tight_layout()
#     plt.show()

# fig, ax = plt.subplots(figsize=(16,9))
# old_df.groupby('node_id')['sm_10cm'].plot(figsize=(16,9), legend=False, ax=ax)
# plt.tight_layout()
# plt.show()
