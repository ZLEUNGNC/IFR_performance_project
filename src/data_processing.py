from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

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
