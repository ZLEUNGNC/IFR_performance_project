import numpy as np
import matplotlib.pyplot as plt


def plot_lap_times(laps):
    plt.plot(
        laps["Lap"],
        laps["D1_lap_times_dropped"],
        label="Driver 1"
    )

    plt.plot(
        laps["Lap"],
        laps["D2_lap_times_dropped"],
        label="Driver 2"
    )

    plt.xlabel("Lap")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.show()


def plot_speed_map(driver1, lap_number):
    lap = driver1[driver1["Lap"] == lap_number]

    min_speed = lap["GPSSpeed (km/h)"].min()
    max_speed = lap["GPSSpeed (km/h)"].max()

    # Convert GPS coordinates from degrees to metres
    lat0 = np.radians(lap["GPSLatitude (deg)"].iloc[0])
    lon0 = np.radians(lap["GPSLongitude (deg)"].iloc[0])

    x = (np.radians(lap["GPSLongitude (deg)"]) - lon0) * 6371000 * np.cos(lat0)
    y = (np.radians(lap["GPSLatitude (deg)"]) - lat0) * 6371000

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

    plt.colorbar(scatter, label="Speed (km/h)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Distance (m)")
    plt.title(f"Driver 1 - Lap {lap_number}")
    plt.axis("equal")
    plt.show()