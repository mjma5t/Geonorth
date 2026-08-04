import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cftime
import datetime
import re

def obs_data_parse(file):
    """
    Parses GWF soil temperature txt files into numpy arrays.
    Handles commented header lines and ISO 8601 timestamps.

    Args:
        file: path to the .txt file

    Returns:
        time:   numpy array of datetime64 timestamps
        values: numpy array of values

    Example:
        time, temp = obs_data_parse("TVC_obs/soil_temp/TVC_SoilTemp_0.05m.txt")
    """
    obs = pd.read_csv(
        file,
        comment="#",
        sep=r"\s+",
        parse_dates=["Timestamp"],
        engine="python"
    )

    time = np.array([
        cftime.DatetimeNoLeap(t.year, t.month, t.day, t.hour, t.minute, t.second)
        for t in pd.DatetimeIndex(obs["Timestamp"])
        if not (t.month == 2 and t.day == 29)
    ])
    values = obs["Value"][~((pd.DatetimeIndex(obs["Timestamp"]).month == 2) & 
                            (pd.DatetimeIndex(obs["Timestamp"]).day == 29))].values.astype(float)
    return time, values


def pull_files(directory_path):
    times = []
    values = []

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        # Check if it is a file (not a directory)
        if os.path.isfile(full_path):
            time, value = obs_data_parse(full_path)
            times.append(time)
            values.append(value)
    return times, values

        

def check_shapes(times):
    shapes = [arr.shape for arr in times]

    if len(set(shapes)) == 1:
        print(f"All arrays match: {shapes[0]}")
    else:
        for i, shape in enumerate(shapes):
            print(f"Index {i}: {shape}")
    return


def make_obs_ds(vari, data_2d, times, site, units, depths):
    ds_obs = xr.Dataset(
        {f"{vari}": (["time", "depth"], data_2d, {"units": f"{units}"})},
        coords={
            "time":  times,
            "depth": depths,
        },
        attrs={"site": f"{site}", "source": "GWF observed"}
    )
    return ds_obs


def pull_files_dupe(directory_path):
    times = []
    values = []
    depths = []
    depth_count = {}  # tracks how many times each base depth has appeared

    for filename in sorted(os.listdir(directory_path)):
        full_path = os.path.join(directory_path, filename)
        if not os.path.isfile(full_path):
            continue

        # Extract depth label e.g. "20", "20B", "20C"
        match = re.search(r'_([0-9]+[A-Z]?)\.txt$', filename, re.IGNORECASE)
        if not match:
            continue

        raw = match.group(1)                        # e.g. "20", "20B", "20C"
        base_depth = float(re.match(r'[0-9]+', raw).group())  # numeric part only
        base_depth_m = base_depth/100


        # Assign decimal offset for duplicates: 20, 20.1, 20.2 ...
        count = depth_count.get(base_depth_m, 0)
        depth = base_depth_m + count * 0.001
        depth_count[base_depth_m] = count + 1

        time, value = obs_data_parse(full_path)
        times.append(time)
        values.append(value)
        depths.append(depth)

    return times, values, depths


def extend_time_and_values(
    time: np.ndarray,
    values: np.ndarray,
    new_start: cftime.DatetimeNoLeap,
    new_end: cftime.DatetimeNoLeap,
    freq: str = "MS"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend a cftime time array and corresponding values array to a new date range,
    filling missing time steps with NaN.

    Parameters
    ----------
    time : np.ndarray
        Original array of cftime.DatetimeNoLeap objects.
    values : np.ndarray
        Observation array with the same length as `time`.
    new_start : cftime.DatetimeNoLeap
        The desired start of the extended time range.
    new_end : cftime.DatetimeNoLeap
        The desired end of the extended time range.
    freq : str
        Frequency for generating the new time axis.
        Supported:
          "MS"  — month start
          "YS"  — year start
          "D"   — daily
          "H"   — hourly
          "30T" — every 30 minutes

    Returns
    -------
    new_time : np.ndarray
        Extended array of cftime.DatetimeNoLeap objects.
    new_values : np.ndarray
        Extended values array, with NaN where no observation exists.
    """
    # --- 1. Build the full target time axis ---
    DELTA_FREQS = {
        "D":   datetime.timedelta(days=1),
        "H":   datetime.timedelta(hours=1),
        "30T": datetime.timedelta(minutes=30),
    }

    if freq == "MS":
        def step(d, n):
            total_months = d.month - 1 + n
            return cftime.DatetimeNoLeap(
                d.year + total_months // 12,
                total_months % 12 + 1,
                1,
                has_year_zero=True
            )
    elif freq == "YS":
        def step(d, n):
            return cftime.DatetimeNoLeap(d.year + n, 1, 1, has_year_zero=True)
    elif freq in DELTA_FREQS:
        delta = DELTA_FREQS[freq]
        def step(d, n):
            return d + delta * n
    else:
        raise ValueError(
            f"Unsupported freq '{freq}'. Choose from: 'MS', 'YS', 'D', 'H', '30T'."
        )

    full_time = []
    n = 0
    while True:
        cursor = step(new_start, n)
        if cursor > new_end:
            break
        full_time.append(cursor)
        n += 1
    new_time = np.array(full_time)

    # --- 2. Build lookup key based on frequency resolution ---
    def to_key(d):
        if freq == "MS":
            return (d.year, d.month)
        elif freq == "YS":
            return (d.year,)
        elif freq == "D":
            return (d.year, d.month, d.day)
        elif freq == "H":
            return (d.year, d.month, d.day, d.hour)
        elif freq == "30T":
            # Normalise minutes to the nearest 30-min slot (0 or 30)
            slot = (d.minute // 30) * 30
            return (d.year, d.month, d.day, d.hour, slot)

    # --- 3. Map original timestamps to positions in the new axis ---
    time_to_index = {to_key(t): i for i, t in enumerate(new_time)}

    # --- 4. Fill values into the new array ---
    new_values = np.full(len(new_time), np.nan)
    for orig_t, orig_v in zip(time, values):
        key = to_key(orig_t)
        if key in time_to_index:
            new_values[time_to_index[key]] = orig_v

    return new_time, new_values


def pull_files_num(directory_path):
    times = []
    values = []
    depths = []

    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        full_path = os.path.join(directory_path, filename)
        if not os.path.isfile(full_path):
            continue

        time, value = obs_data_parse(full_path)
        times.append(time)
        values.append(value)
        depths.append(float(i))

    return times, values, depths

def plot_soil_temp_depths(ds, vari):
    fig, ax = plt.subplots(figsize=(14, 5))
    for depth in ds["depth"].values:
        ax.plot(ds["time"].values, ds[f"{vari}"].sel(depth=depth).values, label=f"{depth}")
    ax.legend(loc="upper right")
    plt.show()

def plot_disconnected(ds, vari):
    fig, ax = plt.subplots(figsize=(14, 5))
    for depth in ds["depth"].values:
        ax.plot(ds["time"].values, ds[f"{vari}"].sel(depth=depth).values, 'o',markersize=2, label=f"{depth}")
    ax.legend(loc="upper right")
    plt.show()