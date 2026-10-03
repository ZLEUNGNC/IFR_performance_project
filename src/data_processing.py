from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data():
    driver1 = read_csv(filepath_or_buffer = "data/D1.csv")
    driver2 = read_csv(filepath_or_buffer = "data/D2.csv")
    laps = read_csv(filepath_or_buffer = "data/Laps.csv")  # loading csv files onto pandas df
    return driver1, driver2, laps

def assign_laps(driver1, driver2, laps):
    lap_times_d1 = laps["D1"].to_numpy()
    lap_times_d2 = laps["D2"].dropna().to_numpy()  # convert to np

    driver1["Lap"] = np.searchsorted(
                lap_times_d1,
                driver1["Time (s)"].to_numpy(),
                side="right") + 1
    driver2["Lap"] = np.searchsorted(
                lap_times_d2,
                driver2["Time (s)"].to_numpy(),
                side="right") + 1  # lap number index

def assign_distance(driver):
    lat = np.radians(driver["GPSLatitude (deg)"])
    lon = np.radians(driver["GPSLongitude (deg)"])

    # Reference point for each lap
    lat0 = lat.groupby(driver["Lap"]).transform("first")
    lon0 = lon.groupby(driver["Lap"]).transform("first")

    R = 6_371_000

    driver["x"] = (lon - lon0) * R * np.cos(lat0)
    driver["y"] = (lat - lat0) * R

    # Distance between consecutive GPS points
    dx = driver["x"].diff()
    dy = driver["y"].diff()

    point_distance = np.sqrt(dx**2 + dy**2)

    # Reset cumulative distance at the start of every lap
    driver["Distance"] = (
        point_distance
        .groupby(driver["Lap"])
        .cumsum()
        .fillna(0)
    )

    return driver