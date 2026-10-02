import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

driver1 = pd.read_csv("data/D1.csv")
driver2 = pd.read_csv("data/D2.csv")
laps = pd.read_csv("data/Laps.csv") #loading csv files onto pandas df

lap_times_D1 = laps["D1"].to_numpy()
lap_times_D2 = laps["D2"].to_numpy() #convert to np

driver1["Lap"] = (np.searchsorted(lap_times_D1,
                                 driver1["Time (s)"].to_numpy(),
                                 side ="right") + 1)
driver2["Lap"] = np.searchsorted(lap_times_D2,
                                 driver2["Time (s)"].to_numpy(),
                                 side = "right") + 1 #lap number index

laps["D1_lap_times"] = laps["D1"].diff()
laps.loc[0, "D1_lap_times"] = laps.loc[0, "D1"]
laps["D2_lap_times"] = laps["D2"].diff()
laps.loc[0, "D2_lap_times"] = laps.loc[0, "D2"] #lap time index

laps["D1_lap_times_dropped"] = laps["D1_lap_times"].iloc[1:-1]
laps["D2_lap_times_dropped"] = laps["D2_lap_times"].iloc[1:-2] #discounting the first and last lap because the data is messy, (D2 has 1 less lap)

plt.plot(laps["Lap"],
         laps["D1_lap_times_dropped"],
         label = "driver1")
plt.plot(laps["Lap"],
         laps["D2_lap_times_dropped"],
         label = "driver2")

plt.xlabel("lap")
plt.ylabel("Lap Time (s)")
plt.legend()
plt.show()


lap_number = int(input("Enter lap number: "))

lap = driver1[driver1["Lap"] == lap_number]

min_speed = lap["GPSSpeed (km/h)"].min()
max_speed = lap["GPSSpeed (km/h)"].max()

# Convert GPS coordinates from degrees to metres
lat0 = np.radians(lap["GPSLatitude (deg)"].iloc[0])
lon0 = np.radians(lap["GPSLongitude (deg)"].iloc[0])

x = (np.radians(lap["GPSLongitude (deg)"]) - lon0) * 6371000 * np.cos(lat0)
y = (np.radians(lap["GPSLatitude (deg)"]) - lat0) * 6371000

# Round colour scale to nearest 10
scale_min = np.floor(min_speed / 10) * 10
scale_max = np.ceil(max_speed / 10) * 10

scatter = plt.scatter(
    x,
    y,
    c=lap["GPSSpeed (km/h)"],
    cmap="turbo",
    vmin=scale_min,
    vmax=scale_max,
    s=3
)

plt.colorbar(
    scatter,
    label="Speed (km/h)"
)

plt.xlabel("Distance (m)")
plt.ylabel("Distance (m)")
plt.title(f"Driver 1 - Lap {lap_number}")
plt.axis("equal")
plt.show()

print(laps["D1_lap_times_dropped"])
print(laps["D2_lap_times_dropped"])
