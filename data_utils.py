from datetime import datetime
import numpy as np


# time intervals marked as driving
INTERVALS = [
    ('24/02/17 07:35:30-+0800', '24/02/17 07:36:20-+0800'),
    ('24/02/17 07:48:00-+0800', '24/02/17 07:49:20-+0800')
]

# sensor results frequency
DATA_fz = 16


def get_sequence_len(seconds):
    """
    Get length of time series sequence
    :param seconds: time equivalent of sequence length
    :return: count of sensor results in sequence length
    """
    return DATA_fz * seconds


def time_to_timestamp(time: str, time_format="%d/%m/%y %H:%M:%S-%z"):
    """
    Convert string time representation to ms
    :param time: time in string format
    :param time_format: regex of time firmat
    :return:
    """
    return datetime.strptime(time, time_format).timestamp() * 1000


TIMESTAMPS_INTERVAL = np.array(
    [[time_to_timestamp(interval[0]), time_to_timestamp(interval[1])] for interval in INTERVALS]
)


def compare(timestamps):
    """
    Generate boolean mask for math certain time intervals
    :param timestamps: Exist timestamps from sensors
    :return: Boolean mask to match hardcoded time intervals
    """
    return np.logical_or(
        np.logical_and(timestamps >= TIMESTAMPS_INTERVAL[0][0], timestamps <= TIMESTAMPS_INTERVAL[0][1]),
        np.logical_and(timestamps >= TIMESTAMPS_INTERVAL[1][0], timestamps <= TIMESTAMPS_INTERVAL[1][1])
    )


def split_series(series, length, pad=1):
    """
    Create time series sequences from full time series
    :param series: Data from sensors
    :param length: count of points from sensors
    :param pad: distance between start indexes of sequences
    :return: array of sensor data with shape (-1, length, features.shape), array of target classes per sequence
    """
    result = []
    target = []
    for i in range(0, series.shape[0], pad):
        if i + length >= series.shape[0]:
            break
        result.append(series[i:i + length, : -1])
        target.append(1 if (1.0 * series[i:i + length, -1].sum() / series[i:i + length, -1].shape[0]) > 0.5 else 0)
    return np.array(result), np.array(target)


def resample_data(X, y):
    """
    Down sample data to condition when it contain same count of pos and neg classes
    :param X: array of time series sequences
    :param y: array of target classes per sequence
    :return: resampled array of time series sequences, resampled array of target classes per sequence
    """
    pos_x = X[y == 1]
    neg_x = X[~(y == 1)]

    down_sample_mask = np.random.choice(neg_x.shape[0], pos_x.shape[0], replace=False)
    neg_x = neg_x[down_sample_mask]
    X = np.concatenate([pos_x, neg_x], axis=0)
    y = np.zeros(X.shape[0])
    y[neg_x.shape[0]:] = 1
    return X, y
