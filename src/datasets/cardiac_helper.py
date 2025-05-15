# import torch
# import numpy as np
# import pandas as pd
# import wfdb
from datetime import datetime, timedelta
import math


## helper functions

def get_start_end_times(record):
    start_dt = datetime.combine(record.base_date, record.base_time)
    duration_sec = record.sig_len / record.fs
    end_dt = start_dt + timedelta(seconds=duration_sec)
    # print("Date:", record.base_date)
    # print("Start time:", record.base_time)
    # print("End time:", end_dt.time())
    return start_dt, end_dt


def get_measurements(record, alarm_ts=None, cutoff_minutes=1, new_fs=50):
    """
    new_fs: the downsampled number of samples taken per second (i.e. Hz)
    """
    signal = record.p_signal[:, 0]  # 1D array of measurements 
    start_time = datetime.combine(record.base_date, record.base_time)

    formatted = []

    # Determine cutoff time in seconds since start
    if alarm_ts is None:
        end_time = start_time + timedelta(seconds=record.sig_len / record.fs)
        cutoff_time = end_time - timedelta(minutes=cutoff_minutes)
    else:
        cutoff_time = alarm_ts - timedelta(minutes=cutoff_minutes)
    total_duration = (cutoff_time - start_time).total_seconds()
    max_idx = min(len(signal), int(total_duration * record.fs))

    step_sz = int(record.fs // new_fs)

    for i in range(0, max_idx, step_sz):
        current_time = start_time + timedelta(seconds=i / record.fs)
        # formatted.append(f"{current_time.time()}, {signal[i]:.6f}")
        formatted.append(signal[i])
    return np.array(formatted), start_time, cutoff_time, new_fs



import random
import copy

# Pipeline:
# 1. get the alarm time
# 2. subtract r = 1- 300 sec from it
# 3. this is your new cutoff time
# 4. then take the preceding 120 seconds from that

# in terms of p_signal data:
# 1. get new cutoff time data = orig_alarm_time_data_index - r * fs
# 2. get new start time data = new_cutoff_time_data_index - 120 * fs


def extract_segment_before_alarm(record, alarm_dt=None, duration_sec=120, max_pred_window_sec=300):
    fs = record.fs
    start_dt = datetime.combine(record.base_date, record.base_time)
    total_duration_sec = record.sig_len / fs
    end_dt = start_dt + timedelta(seconds=total_duration_sec)

    # pick a random cutoff time
    pred_window_sec = random.randint(1, max_pred_window_sec)
    print(pred_window_sec)

    if alarm_dt is not None:
        cutoff_dt = alarm_dt - timedelta(seconds=pred_window_sec) # alarm_dt is guaranteed 
        new_start_dt = cutoff_dt - timedelta(seconds=duration_sec)
        print(start_dt, end_dt)
        print(alarm_dt)
        print(new_start_dt, cutoff_dt)
        assert new_start_dt >= start_dt and cutoff_dt > new_start_dt
    else:
        new_start_dt = start_dt
        cutoff_dt = new_start_dt + timedelta(seconds=duration_sec)

    # get new p_signal
    # print(start_dt, end_dt)
    # print(alarm_dt)
    # print(new_start_dt, cutoff_dt)
    print((new_start_dt - start_dt).total_seconds())
    print((cutoff_dt - new_start_dt).total_seconds())

    start_idx = int((new_start_dt - start_dt).total_seconds()) * fs
    cutoff_idx = math.ceil((cutoff_dt - start_dt).total_seconds()) * fs
    # print(start_idx, cutoff_idx)    
    segment = record.p_signal[start_idx:cutoff_idx]

    # Create new record object with the same metadata
    segment_record = copy.deepcopy(record)
    segment_record.p_signal = segment
    segment_record.sig_len = len(segment)
    segment_record.base_date = new_start_dt.date()
    segment_record.base_time = new_start_dt.time()
    
    return segment_record

