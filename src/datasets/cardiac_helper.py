# import torch
# import numpy as np
# import pandas as pd
# import wfdb
from datetime import datetime, timedelta


## helper functions

def get_start_end_times(record):
    start_dt = datetime.combine(datetime.today(), record.base_time)
    duration_sec = record.sig_len / record.fs
    end_dt = start_dt + timedelta(seconds=duration_sec)
    print("Date:", record.base_date)
    print("Start time:", record.base_time)
    print("End time:", end_dt.time())
    return
    # return start_dt, end_dt


def get_measurements(record, alarm_ts=None, cutoff_minutes=1):
    # Extract relevant fields
    signal = record.p_signal[:, 0]  # 1D array of measurements
    fs = record.fs  # Sampling rate (Hz)
    base_time = record.base_time  # datetime.time
    base_date = record.base_date  # datetime.date
    start_time = datetime.combine(base_date, base_time)
    # Build the list in the format [time: measurement, ...] 
    formatted = []
    # Loop through the signal and build timestamped strings
    for i, val in enumerate(signal):
        current_time = start_time + timedelta(seconds=i / fs)
        if alarm_ts != None:
            cutoff_time = alarm_ts - timedelta(minutes=cutoff_minutes)
            # print("Cutoff time: ", cutoff_time)
            if current_time.time() > cutoff_time.time():
                break
        formatted.append(f"{current_time.time()}, {val:.6f}")
    return formatted