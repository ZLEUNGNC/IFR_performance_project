def calculate_lap_times(laps):
    laps["D1_lap_times"] = laps["D1"].diff()
    laps.loc[0, "D1_lap_times"] = laps.loc[0, "D1"]

    laps["D2_lap_times"] = laps["D2"].diff()
    laps.loc[0, "D2_lap_times"] = laps.loc[0, "D2"]

    laps["D1_lap_times_dropped"] = laps["D1_lap_times"].iloc[1:-1]
    laps["D2_lap_times_dropped"] = laps["D2_lap_times"].iloc[1:-2]

    return laps