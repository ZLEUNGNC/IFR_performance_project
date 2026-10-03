import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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

def plot_speed_map(driverpd, lap_number):
    lap = driverpd[driverpd["Lap"] == lap_number]

    min_speed = lap["GPSSpeed (km/h)"].min()
    max_speed = lap["GPSSpeed (km/h)"].max()

    scale_min = np.floor(min_speed / 10) * 10
    scale_max = np.ceil(max_speed / 10) * 10

    scatter = plt.scatter(
        lap["x"],
        lap["y"],
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

def interactive_track(driverpd, lap_number):

    # Get one lap
    lap = driverpd[driverpd["Lap"] == lap_number].copy()

    x = lap["x"].to_numpy()
    y = lap["y"].to_numpy()
    distance = lap["Distance"].to_numpy()
    speed = lap["GPSSpeed (km/h)"].to_numpy()

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 7))

    # Speed colour scale
    min_speed = np.floor(speed.min() / 10) * 10
    max_speed = np.ceil(speed.max() / 10) * 10

    # Track coloured by speed
    scatter = ax.scatter(
        x,
        y,
        c=speed,
        cmap="turbo",
        vmin=min_speed,
        vmax=max_speed,
        s=8
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="Speed (km/h)"
    )

    # --------------------------------------------------
    # Triangle marker
    # --------------------------------------------------

    triangle = Polygon(
        [
            [x[0], y[0]],
            [x[0], y[0]],
            [x[0], y[0]]
        ],
        closed=True
    )

    ax.add_patch(triangle)

    # --------------------------------------------------
    # Text
    # --------------------------------------------------

    text = ax.text(
        0.02,
        0.95,
        f"Distance: {distance[0]:.2f} m\n"
        f"Speed: {speed[0]:.1f} km/h",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top"
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Driver 1 — Lap {lap_number}")

    dragging = False

    # --------------------------------------------------
    # Find nearest point
    # --------------------------------------------------

    def find_nearest_point(mouse_x, mouse_y):

        d = (
            (x - mouse_x) ** 2
            + (y - mouse_y) ** 2
        )

        return np.argmin(d)

    # --------------------------------------------------
    # Update triangle
    # --------------------------------------------------

    def update_marker(event):

        if event.xdata is None or event.ydata is None:
            return

        index = find_nearest_point(
            event.xdata,
            event.ydata
        )

        # Track direction
        if index == 0:
            dx = x[1] - x[0]
            dy = y[1] - y[0]

        elif index == len(x) - 1:
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]

        else:
            dx = x[index + 1] - x[index - 1]
            dy = y[index + 1] - y[index - 1]

        # Normalise
        length = np.sqrt(dx**2 + dy**2)

        dx /= length
        dy /= length

        # Perpendicular direction
        nx = -dy
        ny = dx

        # Distance of triangle from track
        offset = 3.0

        centre_x = x[index] + nx * offset
        centre_y = y[index] + ny * offset

        # Triangle size
        triangle_length = 3.0
        triangle_width = 1.5

        # Tip points toward track
        tip_x = centre_x - nx * triangle_length
        tip_y = centre_y - ny * triangle_length

        # Base
        base_x = centre_x + nx * triangle_length
        base_y = centre_y + ny * triangle_length

        # Direction across the triangle
        wx = -ny
        wy = nx

        left_x = base_x + wx * triangle_width
        left_y = base_y + wy * triangle_width

        right_x = base_x - wx * triangle_width
        right_y = base_y - wy * triangle_width

        # Move triangle
        triangle.set_xy([
            [tip_x, tip_y],
            [left_x, left_y],
            [right_x, right_y]
        ])

        # Update information
        text.set_text(
            f"Distance: {distance[index]:.2f} m\n"
            f"Speed: {speed[index]:.1f} km/h"
        )

        fig.canvas.draw_idle()

    # --------------------------------------------------
    # Mouse controls
    # --------------------------------------------------

    def on_press(event):

        nonlocal dragging

        if event.inaxes != ax:
            return

        dragging = True
        update_marker(event)

    def on_motion(event):

        if not dragging:
            return

        if event.inaxes != ax:
            return

        update_marker(event)

    def on_release(event):

        nonlocal dragging

        dragging = False

    # Connect mouse events
    fig.canvas.mpl_connect(
        "button_press_event",
        on_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release
    )

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ==========================================================
# 1. CREATE COMMON X/Y COORDINATES
# ==========================================================

def assign_xy(driver, lat0, lon0):

    lat = np.radians(driver["GPSLatitude (deg)"])
    lon = np.radians(driver["GPSLongitude (deg)"])

    R = 6_371_000

    driver["x"] = (
        lon - lon0
    ) * R * np.cos(lat0)

    driver["y"] = (
        lat - lat0
    ) * R

    return driver


# ==========================================================
# 2. CREATE DISTANCE ALONG EACH DRIVER'S TRACK
# ==========================================================

def assign_distance(driver):

    dx = driver.groupby("Lap")["x"].diff()
    dy = driver.groupby("Lap")["y"].diff()

    point_distance = np.sqrt(dx**2 + dy**2)

    driver["Distance"] = (point_distance.groupby(driver["Lap"]).cumsum().fillna(0))

    return driver


# ==========================================================
# 3. INTERACTIVE DRIVER COMPARISON
# ==========================================================

def compare_drivers(driver1, driver2, lap_number):

    lat0 = np.radians(
        driver1["GPSLatitude (deg)"].iloc[0]
    )

    lon0 = np.radians(
        driver1["GPSLongitude (deg)"].iloc[0]
    )

    driver1 = assign_xy(
        driver1,
        lat0,
        lon0
    )

    driver2 = assign_xy(
        driver2,
        lat0,
        lon0
    )

    driver1 = assign_distance(driver1)
    driver2 = assign_distance(driver2)

    # ------------------------------------------------------
    # Get lap
    # ------------------------------------------------------

    lap1 = driver1[
        driver1["Lap"] == lap_number
    ].copy()

    lap2 = driver2[
        driver2["Lap"] == lap_number
    ].copy()

    # Driver 1
    x1 = lap1["x"].to_numpy()
    y1 = lap1["y"].to_numpy()
    d1 = lap1["Distance"].to_numpy()
    s1 = lap1["GPSSpeed (km/h)"].to_numpy()

    # Driver 2
    x2 = lap2["x"].to_numpy()
    y2 = lap2["y"].to_numpy()
    d2 = lap2["Distance"].to_numpy()
    s2 = lap2["GPSSpeed (km/h)"].to_numpy()

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

    # ------------------------------------------------------
    # Speed colour scale
    # ------------------------------------------------------

    min_speed = min(
        np.floor(s1.min() / 10) * 10,
        np.floor(s2.min() / 10) * 10
    )

    max_speed = max(
        np.ceil(s1.max() / 10) * 10,
        np.ceil(s2.max() / 10) * 10
    )

    # ------------------------------------------------------
    # Driver 1 track
    # ------------------------------------------------------

    scatter1 = ax1.scatter(
        x1,
        y1,
        c=s1,
        cmap="turbo",
        vmin=min_speed,
        vmax=max_speed,
        s=8
    )

    # ------------------------------------------------------
    # Driver 2 track
    # ------------------------------------------------------

    ax2.scatter(
        x2,
        y2,
        c=s2,
        cmap="turbo",
        vmin=min_speed,
        vmax=max_speed,
        s=8
    )

    # ------------------------------------------------------
    # Shared colourbar
    # ------------------------------------------------------

    fig.colorbar(
        scatter1,
        ax=[ax1, ax2],
        label="Speed (km/h)"
    )

    # ------------------------------------------------------
    # Triangle markers
    # ------------------------------------------------------

    triangle1 = Polygon(
        [
            [x1[0], y1[0]],
            [x1[0], y1[0]],
            [x1[0], y1[0]]
        ],
        closed=True
    )

    triangle2 = Polygon(
        [
            [x2[0], y2[0]],
            [x2[0], y2[0]],
            [x2[0], y2[0]]
        ],
        closed=True
    )

    ax1.add_patch(triangle1)
    ax2.add_patch(triangle2)

    # ------------------------------------------------------
    # Information text
    # ------------------------------------------------------

    text1 = ax1.text(
        0.02,
        0.95,
        "",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top"
    )

    text2 = ax2.text(
        0.02,
        0.95,
        "",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top"
    )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------

    ax1.set_title(
        f"Driver 1 — Lap {lap_number}"
    )

    ax2.set_title(
        f"Driver 2 — Lap {lap_number}"
    )

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # ------------------------------------------------------
    # Find nearest track point
    # ------------------------------------------------------

    def find_nearest(
        x,
        y,
        mouse_x,
        mouse_y
    ):

        distances = (
            (x - mouse_x) ** 2
            + (y - mouse_y) ** 2
        )

        return np.argmin(distances)

    # ------------------------------------------------------
    # Move triangle
    # ------------------------------------------------------

    def move_triangle(
        triangle,
        x,
        y,
        index
    ):

        # Find direction of track
        if index == 0:

            dx = x[1] - x[0]
            dy = y[1] - y[0]

        elif index == len(x) - 1:

            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]

        else:

            dx = x[index + 1] - x[index - 1]
            dy = y[index + 1] - y[index - 1]

        # Normalise
        length = np.sqrt(
            dx**2 + dy**2
        )

        dx /= length
        dy /= length

        # Perpendicular vector
        nx = -dy
        ny = dx

        # Distance from track
        offset = 3.0

        centre_x = (
            x[index]
            + nx * offset
        )

        centre_y = (
            y[index]
            + ny * offset
        )

        # Triangle size
        triangle_length = 3.0
        triangle_width = 1.5

        # Tip points toward track
        tip_x = (
            centre_x
            - nx * triangle_length
        )

        tip_y = (
            centre_y
            - ny * triangle_length
        )

        # Base
        base_x = (
            centre_x
            + nx * triangle_length
        )

        base_y = (
            centre_y
            + ny * triangle_length
        )

        # Width direction
        wx = -ny
        wy = nx

        left_x = (
            base_x
            + wx * triangle_width
        )

        left_y = (
            base_y
            + wy * triangle_width
        )

        right_x = (
            base_x
            - wx * triangle_width
        )

        right_y = (
            base_y
            - wy * triangle_width
        )

        triangle.set_xy([
            [tip_x, tip_y],
            [left_x, left_y],
            [right_x, right_y]
        ])

    # ------------------------------------------------------
    # Synchronise drivers using X/Y position
    # ------------------------------------------------------

    def update_from_d1(index1):

        # Physical position on D1 track
        target_x = x1[index1]
        target_y = y1[index1]

        # Find closest physical point on D2 track
        index2 = find_nearest(
            x2,
            y2,
            target_x,
            target_y
        )

        # Move triangles
        move_triangle(
            triangle1,
            x1,
            y1,
            index1
        )

        move_triangle(
            triangle2,
            x2,
            y2,
            index2
        )

        # Update text
        text1.set_text(
            f"Distance: {d1[index1]:.2f} m\n"
            f"Speed: {s1[index1]:.1f} km/h"
        )

        text2.set_text(
            f"Distance: {d2[index2]:.2f} m\n"
            f"Speed: {s2[index2]:.1f} km/h"
        )

        fig.canvas.draw_idle()

    # ------------------------------------------------------

    def update_from_d2(index2):

        # Physical position on D2 track
        target_x = x2[index2]
        target_y = y2[index2]

        # Find closest physical point on D1 track
        index1 = find_nearest(
            x1,
            y1,
            target_x,
            target_y
        )

        # Move triangles
        move_triangle(
            triangle1,
            x1,
            y1,
            index1
        )

        move_triangle(
            triangle2,
            x2,
            y2,
            index2
        )

        # Update text
        text1.set_text(
            f"Distance: {d1[index1]:.2f} m\n"
            f"Speed: {s1[index1]:.1f} km/h"
        )

        text2.set_text(
            f"Distance: {d2[index2]:.2f} m\n"
            f"Speed: {s2[index2]:.1f} km/h"
        )

        fig.canvas.draw_idle()

    # ------------------------------------------------------
    # Mouse controls
    # ------------------------------------------------------

    dragging = False

    def on_press(event):

        nonlocal dragging

        if event.inaxes not in [ax1, ax2]:
            return

        dragging = True

        if event.inaxes == ax1:

            index1 = find_nearest(
                x1,
                y1,
                event.xdata,
                event.ydata
            )

            update_from_d1(index1)

        else:

            index2 = find_nearest(
                x2,
                y2,
                event.xdata,
                event.ydata
            )

            update_from_d2(index2)

    # ------------------------------------------------------

    def on_motion(event):

        if not dragging:
            return

        if event.inaxes not in [ax1, ax2]:
            return

        if event.inaxes == ax1:

            index1 = find_nearest(
                x1,
                y1,
                event.xdata,
                event.ydata
            )

            update_from_d1(index1)

        else:

            index2 = find_nearest(
                x2,
                y2,
                event.xdata,
                event.ydata
            )

            update_from_d2(index2)

    # ------------------------------------------------------

    def on_release(event):

        nonlocal dragging

        dragging = False

    # ------------------------------------------------------
    # Connect mouse events
    # ------------------------------------------------------

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release
    )

    # ------------------------------------------------------
    # Start at beginning
    # ------------------------------------------------------

    update_from_d1(0)

    plt.show()

