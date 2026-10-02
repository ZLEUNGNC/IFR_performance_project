from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data():
    driver1 = read_csv("data/D1.csv")
    driver2 = read_csv("data/D2.csv")
    laps = read_csv("data/Laps.csv")  # loading csv files onto pandas df
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

def gps_conv(driverpd, lap_number):
    lap = driverpd[driverpd["Lap"] == lap_number]

    lat0 = np.radians(lap["GPSLatitude (deg)"].iloc[0])
    lon0 = np.radians(lap["GPSLongitude (deg)"].iloc[0])

    x = (np.radians(lap["GPSLongitude (deg)"]) - lon0) * 6371000 * np.cos(lat0)
    y = (np.radians(lap["GPSLatitude (deg)"]) - lat0) * 6371000
    return x, y

def cumdist(driverpd, lap_number):
    x,y = gps_conv(driverpd, lap_number)
    xy = zip(x,y)

    print(xy)

load_data()
cumdist(driver1, 3)

    # # Define two points as tuples or lists
    # point1 = (1, 2)
    # point2 = (4, 6)
    #
    # # Calculate Euclidean distance
    # distance = math.dist(point1, point2)
    #
    # driverpd["distance"] = math.dist(point1, xy[])