import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
import numpy as np

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

    # Figure
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

    # Triangle marker

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

    # ======================================================
    # COMMON GPS COORDINATE SYSTEM
    # ======================================================

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

    # ======================================================
    # GET LAP
    # ======================================================

    lap1 = driver1[
        driver1["Lap"] == lap_number
    ].copy()

    lap2 = driver2[
        driver2["Lap"] == lap_number
    ].copy()

    if lap1.empty or lap2.empty:
        raise ValueError(
            f"Lap {lap_number} is missing from one of the drivers."
        )

    # ======================================================
    # BASIC DATA
    # ======================================================

    x1 = lap1["x"].to_numpy()
    y1 = lap1["y"].to_numpy()
    d1 = lap1["Distance"].to_numpy()
    s1 = lap1["GPSSpeed (km/h)"].to_numpy()

    x2 = lap2["x"].to_numpy()
    y2 = lap2["y"].to_numpy()
    d2 = lap2["Distance"].to_numpy()
    s2 = lap2["GPSSpeed (km/h)"].to_numpy()

    t1 = lap1["Time (s)"].to_numpy()
    t2 = lap2["Time (s)"].to_numpy()

    t1_rel = t1 - t1[0]
    t2_rel = t2 - t2[0]

    # ======================================================
    # TELEMETRY COLUMNS
    # ======================================================

    LAT_ACC_COL = "GPSLatAcc (g)"
    LON_ACC_COL = "GPSLonAcc (g)"

    def find_column(df, words, name):

        candidates = []

        for column in df.columns:

            column_lower = column.lower()

            if all(
                word.lower() in column_lower
                for word in words
            ):
                candidates.append(column)

        if len(candidates) == 0:
            raise KeyError(
                f"Could not find {name} column.\n\n"
                f"Available columns:\n{df.columns.tolist()}"
            )

        return candidates[0]

    brake_col1 = find_column(
        lap1,
        ["brake", "pressure"],
        "brake pressure"
    )

    brake_col2 = find_column(
        lap2,
        ["brake", "pressure"],
        "brake pressure"
    )

    throttle_col1 = find_column(
        lap1,
        ["throttle"],
        "throttle"
    )

    throttle_col2 = find_column(
        lap2,
        ["throttle"],
        "throttle"
    )

    # print(
    #     f"D1 brake column: {brake_col1}"
    # )
    #
    # print(
    #     f"D2 brake column: {brake_col2}"
    # )
    #
    # print(
    #     f"D1 throttle column: {throttle_col1}"
    # )
    #
    # print(
    #     f"D2 throttle column: {throttle_col2}"
    # )

    # ======================================================
    # UNIT CONVERSION
    # ======================================================

    def pressure_to_pa(series, column):

        name = column.lower()

        if "mpa" in name:
            return series * 1_000_000

        if "kpa" in name:
            return series * 1_000

        if "bar" in name:
            return series * 100_000

        if "psi" in name:
            return series * 6894.757

        return series

    brake1 = pressure_to_pa(
        lap1[brake_col1].to_numpy(),
        brake_col1
    )

    brake2 = pressure_to_pa(
        lap2[brake_col2].to_numpy(),
        brake_col2
    )

    def throttle_to_percent(series):

        if np.nanmax(
            np.abs(series)
        ) <= 1.05:

            return series * 100

        return series

    throttle1 = throttle_to_percent(
        lap1[throttle_col1].to_numpy()
    )

    throttle2 = throttle_to_percent(
        lap2[throttle_col2].to_numpy()
    )

    # ======================================================
    # ACCELERATION
    # ======================================================

    lat_acc1_g = lap1[
        LAT_ACC_COL
    ].to_numpy()

    lat_acc2_g = lap2[
        LAT_ACC_COL
    ].to_numpy()

    lon_acc1_g = lap1[
        LON_ACC_COL
    ].to_numpy()

    lon_acc2_g = lap2[
        LON_ACC_COL
    ].to_numpy()

    g = 9.80665

    lon_acc1 = lon_acc1_g * g
    lon_acc2 = lon_acc2_g * g

    # ======================================================
    # DRIVER COLOURS
    # ======================================================

    D1_COLOUR = "tab:blue"
    D2_COLOUR = "tab:orange"

    # ======================================================
    # SPEED COLOUR SCALE
    # ======================================================

    min_speed = min(
        np.floor(s1.min() / 10) * 10,
        np.floor(s2.min() / 10) * 10
    )

    max_speed = max(
        np.ceil(s1.max() / 10) * 10,
        np.ceil(s2.max() / 10) * 10
    )

    speed_norm = Normalize(
        vmin=min_speed,
        vmax=max_speed
    )

    # ======================================================
    # FIND NEAREST POINT
    # ======================================================

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

        return np.argmin(
            distances
        )

    # ======================================================
    # PHYSICAL MATCHING
    # ======================================================

    match2_for_d1 = np.array([
        find_nearest(
            x2,
            y2,
            x,
            y
        )
        for x, y in zip(
            x1,
            y1
        )
    ])

    match1_for_d2 = np.array([
        find_nearest(
            x1,
            y1,
            x,
            y
        )
        for x, y in zip(
            x2,
            y2
        )
    ])

    # ======================================================
    # D2 - D1 TIME DIFFERENCE
    # ======================================================

    # Remove duplicate distance values from D2
    d2_unique, unique_indices = np.unique(
        d2,
        return_index=True
    )

    t2_unique = t2_rel[
        unique_indices
    ]

    # Find D2's elapsed time at each D1 distance
    d2_time_at_d1 = np.interp(
        d1,
        d2_unique,
        t2_unique
    )

    # D2 - D1
    delta_time = (
            d2_time_at_d1
            - t1_rel
    )

    # Both drivers start at zero
    delta_time -= delta_time[0]

    # ======================================================
    # ROOT WINDOW
    # ======================================================

    root = tk.Tk()

    root.title(
        f"Formula Student Driver Comparison - Lap {lap_number}"
    )

    root.geometry(
        "1800x1000"
    )

    root.minsize(
        1200,
        700
    )

    # ======================================================
    # MAIN LAYOUT
    #
    # EXACTLY:
    #
    # LEFT  = 1/3
    # RIGHT = 2/3
    # ======================================================

    root.grid_rowconfigure(
        0,
        weight=1
    )

    root.grid_columnconfigure(
        0,
        weight=1,
        uniform="main"
    )

    root.grid_columnconfigure(
        1,
        weight=2,
        uniform="main"
    )

    # ======================================================
    # LEFT FIXED PANEL
    # ======================================================

    left_frame = tk.Frame(
        root,
        bd=0,
        highlightthickness=0
    )

    left_frame.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    # ======================================================
    # RIGHT SCROLLABLE PANEL
    # ======================================================

    right_frame = tk.Frame(
        root,
        bd=0,
        highlightthickness=0
    )

    right_frame.grid(
        row=0,
        column=1,
        sticky="nsew"
    )

    right_frame.grid_rowconfigure(
        0,
        weight=1
    )

    right_frame.grid_columnconfigure(
        0,
        weight=1
    )

    right_canvas = tk.Canvas(
        right_frame,
        highlightthickness=0,
        bd=0
    )

    right_scrollbar = tk.Scrollbar(
        right_frame,
        orient="vertical",
        command=right_canvas.yview
    )

    right_canvas.configure(
        yscrollcommand=right_scrollbar.set
    )

    right_canvas.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    right_scrollbar.grid(
        row=0,
        column=1,
        sticky="ns"
    )

    right_content = tk.Frame(
        right_canvas,
        bd=0,
        highlightthickness=0
    )

    right_window = right_canvas.create_window(
        (0, 0),
        window=right_content,
        anchor="nw"
    )

    def update_scroll_region(event=None):

        right_canvas.configure(
            scrollregion=right_canvas.bbox("all")
        )

    right_content.bind(
        "<Configure>",
        update_scroll_region
    )

    def resize_right_content(event):

        right_canvas.itemconfigure(
            right_window,
            width=event.width
        )

    right_canvas.bind(
        "<Configure>",
        resize_right_content
    )

    # ======================================================
    # RIGHT MOUSE WHEEL
    # ======================================================

    def scroll_right(event):

        if (
            hasattr(event, "delta")
            and event.delta != 0
        ):

            right_canvas.yview_scroll(
                int(
                    -event.delta / 120
                ),
                "units"
            )

        return "break"

    def scroll_right_up(event):

        right_canvas.yview_scroll(
            -3,
            "units"
        )

        return "break"

    def scroll_right_down(event):

        right_canvas.yview_scroll(
            3,
            "units"
        )

        return "break"

    # ======================================================
    # LEFT FIGURE
    #
    # TRACKS ≈ 5/8
    # G-G ≈ 3/16
    # LAT G/SPEED ≈ 3/16
    #
    # Extremely small margins.
    # ======================================================

    left_fig = plt.Figure(
        figsize=(7.0, 12.0),
        dpi=100
    )

    left_gs = left_fig.add_gridspec(
        3,
        2,

        height_ratios=[
            3.3,
            1.0,
            1.0
        ],

        hspace=0.22,
        wspace=0.10,

        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.015
    )

    # ======================================================
    # TRACK AXES
    # ======================================================

    ax1 = left_fig.add_subplot(
        left_gs[0, 0]
    )

    ax2 = left_fig.add_subplot(
        left_gs[0, 1]
    )

    # ======================================================
    # G-G AXES
    # ======================================================

    ax_gg1 = left_fig.add_subplot(
        left_gs[1, 0]
    )

    ax_gg2 = left_fig.add_subplot(
        left_gs[1, 1]
    )

    # ======================================================
    # LATERAL G VS SPEED AXES
    # ======================================================

    ax_lat_speed1 = left_fig.add_subplot(
        left_gs[2, 0]
    )

    ax_lat_speed2 = left_fig.add_subplot(
        left_gs[2, 1]
    )

    # ======================================================
    # RIGHT FIGURE
    # ======================================================

    right_fig = plt.Figure(
        figsize=(12.5, 34),
        dpi=100
    )

    right_gs = right_fig.add_gridspec(
        6,
        1,
        height_ratios=[
            1,
            1,
            1,
            1,
            1,
            1
        ],
        hspace=0.45,
        left=0.07,
        right=0.985,
        top=0.99,
        bottom=0.02
    )

    ax_speed = right_fig.add_subplot(
        right_gs[0, 0]
    )

    ax_brake = right_fig.add_subplot(
        right_gs[1, 0]
    )

    ax_throttle = right_fig.add_subplot(
        right_gs[2, 0]
    )

    ax_long_distance = right_fig.add_subplot(
        right_gs[3, 0]
    )

    ax_lat_distance = right_fig.add_subplot(
        right_gs[5, 0]
    )

    ax_delta = right_fig.add_subplot(
        right_gs[4, 0]
    )

    # ======================================================
    # EMBED LEFT FIGURE
    # ======================================================

    left_mpl = FigureCanvasTkAgg(
        left_fig,
        master=left_frame
    )

    left_mpl.draw()

    left_mpl_widget = (
        left_mpl.get_tk_widget()
    )

    left_mpl_widget.pack(
        fill="both",
        expand=True,
        padx=0,
        pady=0
    )

    # ======================================================
    # EMBED RIGHT FIGURE
    # ======================================================

    right_mpl = FigureCanvasTkAgg(
        right_fig,
        master=right_content
    )

    right_mpl.draw()

    right_mpl_widget = (
        right_mpl.get_tk_widget()
    )

    right_mpl_widget.pack(
        fill="both",
        expand=True,
        padx=0,
        pady=0
    )

    # ======================================================
    # RIGHT SCROLL EVENTS
    # ======================================================

    right_mpl_widget.bind(
        "<MouseWheel>",
        scroll_right
    )

    right_mpl_widget.bind(
        "<Button-4>",
        scroll_right_up
    )

    right_mpl_widget.bind(
        "<Button-5>",
        scroll_right_down
    )

    right_canvas.bind(
        "<MouseWheel>",
        scroll_right
    )

    right_canvas.bind(
        "<Button-4>",
        scroll_right_up
    )

    right_canvas.bind(
        "<Button-5>",
        scroll_right_down
    )

    # ======================================================
    # TRACK SPEED COLOURING
    # ======================================================

    scatter1 = ax1.scatter(
        x1,
        y1,
        c=s1,
        cmap="turbo",
        norm=speed_norm,
        s=5
    )

    scatter2 = ax2.scatter(
        x2,
        y2,
        c=s2,
        cmap="turbo",
        norm=speed_norm,
        s=5
    )

    # ======================================================
    # TRACK FORMATTING
    # ======================================================

    ax1.set_title(
        "D1 Track",
        fontsize=12,
        pad=3
    )

    ax2.set_title(
        "D2 Track",
        fontsize=12,
        pad=3
    )

    ax1.set_aspect(
        "equal",
        adjustable="box"
    )

    ax2.set_aspect(
        "equal",
        adjustable="box"
    )

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_xticks([])
    ax2.set_yticks([])

    # ------------------------------------------------------
    # Tight track limits
    # ------------------------------------------------------

    x_range1 = max(
        x1.max() - x1.min(),
        1
    )

    y_range1 = max(
        y1.max() - y1.min(),
        1
    )

    x_range2 = max(
        x2.max() - x2.min(),
        1
    )

    y_range2 = max(
        y2.max() - y2.min(),
        1
    )

    # Only 2.5% padding around the actual track

    ax1.set_xlim(
        x1.min() - 0.025 * x_range1,
        x1.max() + 0.025 * x_range1
    )

    ax1.set_ylim(
        y1.min() - 0.025 * y_range1,
        y1.max() + 0.025 * y_range1
    )

    ax2.set_xlim(
        x2.min() - 0.025 * x_range2,
        x2.max() + 0.025 * x_range2
    )

    ax2.set_ylim(
        y2.min() - 0.025 * y_range2,
        y2.max() + 0.025 * y_range2
    )

    # ======================================================
    # TRIANGLES
    # ======================================================

    triangle1 = Polygon(
        [
            [x1[0], y1[0]],
            [x1[0], y1[0]],
            [x1[0], y1[0]]
        ],
        closed=True,
        facecolor=D1_COLOUR,
        edgecolor="black",
        zorder=20
    )

    triangle2 = Polygon(
        [
            [x2[0], y2[0]],
            [x2[0], y2[0]],
            [x2[0], y2[0]]
        ],
        closed=True,
        facecolor=D2_COLOUR,
        edgecolor="black",
        zorder=20
    )

    ax1.add_patch(
        triangle1
    )

    ax2.add_patch(
        triangle2
    )

    # ======================================================
    # TRACK INFORMATION
    # ======================================================

    text1 = ax1.text(
        0.98,
        0.98,
        "",
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            alpha=0.8
        ),
        zorder=30
    )

    text2 = ax2.text(
        0.98,
        0.98,
        "",
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            alpha=0.8
        ),
        zorder=30
    )

    # ======================================================
    # G-G FORMATTING
    # ======================================================

    for ax, title in [
        (ax_gg1, "D1 G-G"),
        (ax_gg2, "D2 G-G")
    ]:

        ax.set_title(
            title,
            fontsize=10,
            pad=2
        )

        ax.set_xlabel(
            "Lateral G",
            fontsize=8,
            labelpad=1
        )

        ax.set_ylabel(
            "Longitudinal G",
            fontsize=8,
            labelpad=1
        )

        ax.tick_params(
            labelsize=7,
            pad=1
        )

        ax.grid(
            True,
            alpha=0.2
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.7,
            alpha=0.4
        )

        ax.axvline(
            0,
            color="black",
            linewidth=0.7,
            alpha=0.4
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

    # ======================================================
    # G-G LIMITS
    # ======================================================

    gg_max = np.nanmax(
        np.abs(
            np.concatenate([
                lat_acc1_g,
                lat_acc2_g,
                lon_acc1_g,
                lon_acc2_g
            ])
        )
    )

    if (
        not np.isfinite(gg_max)
        or gg_max == 0
    ):

        gg_max = 1.0

    gg_max *= 1.10

    ax_gg1.set_xlim(
        -gg_max,
        gg_max
    )

    ax_gg1.set_ylim(
        -gg_max,
        gg_max
    )

    ax_gg2.set_xlim(
        -gg_max,
        gg_max
    )

    ax_gg2.set_ylim(
        -gg_max,
        gg_max
    )

    # ======================================================
    # FULL G-G TRACE
    # ======================================================

    ax_gg1.plot(
        lat_acc1_g,
        lon_acc1_g,
        color=D1_COLOUR,
        alpha=0.10,
        linewidth=1
    )

    ax_gg2.plot(
        lat_acc2_g,
        lon_acc2_g,
        color=D2_COLOUR,
        alpha=0.10,
        linewidth=1
    )

    # ======================================================
    # DECAYING G-G TRAILS
    # ======================================================

    gg_trail1 = LineCollection(
        [],
        linewidths=1.8,
        zorder=5
    )

    gg_trail2 = LineCollection(
        [],
        linewidths=1.8,
        zorder=5
    )

    ax_gg1.add_collection(
        gg_trail1
    )

    ax_gg2.add_collection(
        gg_trail2
    )

    # ======================================================
    # G-G CURRENT POINTS
    # ======================================================

    gg_marker1, = ax_gg1.plot(
        [lat_acc1_g[0]],
        [lon_acc1_g[0]],
        marker="o",
        markersize=4,
        color=D1_COLOUR,
        linestyle="",
        zorder=10
    )

    gg_marker2, = ax_gg2.plot(
        [lat_acc2_g[0]],
        [lon_acc2_g[0]],
        marker="o",
        markersize=4,
        color=D2_COLOUR,
        linestyle="",
        zorder=10
    )

    # ======================================================
    # LATERAL G VS SPEED
    # ======================================================

    ax_lat_speed1.scatter(
        s1,
        lat_acc1_g,
        s=5,
        color=D1_COLOUR,
        alpha=0.20
    )

    ax_lat_speed2.scatter(
        s2,
        lat_acc2_g,
        s=5,
        color=D2_COLOUR,
        alpha=0.20
    )

    for ax, title in [
        (
            ax_lat_speed1,
            "D1 Lateral G vs Speed"
        ),
        (
            ax_lat_speed2,
            "D2 Lateral G vs Speed"
        )
    ]:

        ax.axhline(
            0,
            color="black",
            linewidth=0.7,
            alpha=0.4
        )

        ax.set_title(
            title,
            fontsize=10,
            pad=2
        )

        ax.set_xlabel(
            "Speed (km/h)",
            fontsize=8,
            labelpad=1
        )

        ax.set_ylabel(
            "Lateral G",
            fontsize=8,
            labelpad=1
        )

        ax.tick_params(
            labelsize=7,
            pad=1
        )

        ax.grid(
            True,
            alpha=0.2
        )

    # ======================================================
    # SPEED VS DISTANCE
    # ======================================================

    ax_speed.plot(
        d1,
        s1,
        color=D1_COLOUR,
        label="Driver 1"
    )

    ax_speed.plot(
        d2,
        s2,
        color=D2_COLOUR,
        label="Driver 2"
    )

    ax_speed.set_title(
        "Speed vs Distance"
    )

    ax_speed.set_xlabel(
        "Distance (m)"
    )

    ax_speed.set_ylabel(
        "Speed (km/h)"
    )

    ax_speed.grid(
        True,
        alpha=0.3
    )

    ax_speed.legend()

    # ======================================================
    # BRAKE PRESSURE VS DISTANCE
    # ======================================================

    ax_brake.plot(
        d1,
        brake1,
        color=D1_COLOUR,
        label="Driver 1"
    )

    ax_brake.plot(
        d2,
        brake2,
        color=D2_COLOUR,
        label="Driver 2"
    )

    ax_brake.set_title(
        "Brake Pressure vs Distance"
    )

    ax_brake.set_xlabel(
        "Distance (m)"
    )

    ax_brake.set_ylabel(
        "Brake Pressure (Pa)"
    )

    ax_brake.grid(
        True,
        alpha=0.3
    )

    ax_brake.legend()

    # ======================================================
    # THROTTLE VS DISTANCE
    # ======================================================

    ax_throttle.plot(
        d1,
        throttle1,
        color=D1_COLOUR,
        label="Driver 1"
    )

    ax_throttle.plot(
        d2,
        throttle2,
        color=D2_COLOUR,
        label="Driver 2"
    )

    ax_throttle.set_title(
        "Throttle vs Distance"
    )

    ax_throttle.set_xlabel(
        "Distance (m)"
    )

    ax_throttle.set_ylabel(
        "Throttle (%)"
    )

    ax_throttle.set_ylim(
        -10,
        105
    )

    ax_throttle.grid(
        True,
        alpha=0.3
    )

    ax_throttle.legend()

    # ======================================================
    # LONGITUDINAL ACCELERATION VS DISTANCE
    # ======================================================

    ax_long_distance.plot(
        d1,
        lon_acc1,
        color=D1_COLOUR,
        label="Driver 1"
    )

    ax_long_distance.plot(
        d2,
        lon_acc2,
        color=D2_COLOUR,
        label="Driver 2"
    )

    ax_long_distance.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.4
    )

    ax_long_distance.set_title(
        "Longitudinal G vs Distance"
    )

    ax_long_distance.set_xlabel(
        "Distance (m)"
    )

    ax_long_distance.set_ylabel(
        "Longitudinal acceleration (g)"
    )

    ax_long_distance.grid(
        True,
        alpha=0.3
    )

    ax_long_distance.legend()

    # ======================================================
    # D2 - D1 TIME
    # ======================================================

    ax_delta.plot(
        d1,
        delta_time,
        color="black"
    )

    ax_delta.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.4
    )

    ax_delta.set_title(
        "D2 − D1 Time Difference"
    )

    ax_delta.set_xlabel(
        "Distance (m)"
    )

    ax_delta.set_ylabel(
        "Time difference (s)"
    )

    ax_delta.grid(
        True,
        alpha=0.3
    )

    # ======================================================
    # LATERAL G VS DISTANCE
    # ======================================================

    ax_lat_distance.plot(
        d1,
        lat_acc1_g,
        color=D1_COLOUR,
        label="Driver 1"
    )

    ax_lat_distance.plot(
        d2,
        lat_acc2_g,
        color=D2_COLOUR,
        label="Driver 2"
    )

    ax_lat_distance.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.4
    )

    ax_lat_distance.set_title(
        "Lateral G vs Distance"
    )

    ax_lat_distance.set_xlabel(
        "Distance (m)"
    )

    ax_lat_distance.set_ylabel(
        "Lateral acceleration (g)"
    )

    ax_lat_distance.grid(
        True,
        alpha=0.3
    )

    ax_lat_distance.legend()

    # ======================================================
    # POINT LABEL
    # ======================================================

    def make_point_label(
        ax,
        colour,
        text
    ):

        return ax.annotate(
            text,
            xy=(0, 0),
            xycoords="data",
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=colour,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor=colour,
                alpha=0.85
            ),
            zorder=30
        )

    # ======================================================
    # SPEED LABELS
    # ======================================================

    speed_label1 = make_point_label(
        ax_speed,
        D1_COLOUR,
        ""
    )

    speed_label2 = make_point_label(
        ax_speed,
        D2_COLOUR,
        ""
    )

    # ======================================================
    # BRAKE LABELS
    # ======================================================

    brake_label1 = make_point_label(
        ax_brake,
        D1_COLOUR,
        ""
    )

    brake_label2 = make_point_label(
        ax_brake,
        D2_COLOUR,
        ""
    )

    # ======================================================
    # THROTTLE LABELS
    # ======================================================

    throttle_label1 = make_point_label(
        ax_throttle,
        D1_COLOUR,
        ""
    )

    throttle_label2 = make_point_label(
        ax_throttle,
        D2_COLOUR,
        ""
    )

    # ======================================================
    # LONG ACCEL LABELS
    # ======================================================

    accel_label1 = make_point_label(
        ax_long_distance,
        D1_COLOUR,
        ""
    )

    accel_label2 = make_point_label(
        ax_long_distance,
        D2_COLOUR,
        ""
    )

    # ======================================================
    # DELTA LABEL
    # ======================================================

    delta_label = make_point_label(
        ax_delta,
        "black",
        ""
    )

    # ======================================================
    # LATERAL G LABELS
    # ======================================================

    lat_label1 = make_point_label(
        ax_lat_distance,
        D1_COLOUR,
        ""
    )

    lat_label2 = make_point_label(
        ax_lat_distance,
        D2_COLOUR,
        ""
    )

    # ======================================================
    # MASTER UPDATE
    # ======================================================

    def update_position(index1):

        # --------------------------------------------------
        # MATCH D2 PHYSICAL POSITION
        # --------------------------------------------------

        index2 = match2_for_d1[
            index1
        ]

        # --------------------------------------------------
        # TRACKS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # TRACK TEXT
        # --------------------------------------------------

        text1.set_text(
            f"D1\n"
            f"{d1[index1]:.1f} m\n"
            f"{s1[index1]:.1f} km/h"
        )

        text2.set_text(
            f"D2\n"
            f"{d2[index2]:.1f} m\n"
            f"{s2[index2]:.1f} km/h"
        )

        # --------------------------------------------------
        # SPEED
        # --------------------------------------------------

        speed_label1.xy = (
            d1[index1],
            s1[index1]
        )

        speed_label1.set_text(
            f"D1  {s1[index1]:.1f} km/h"
        )

        speed_label2.xy = (
            d2[index2],
            s2[index2]
        )

        speed_label2.set_text(
            f"D2  {s2[index2]:.1f} km/h"
        )

        # --------------------------------------------------
        # BRAKE
        # --------------------------------------------------

        brake_label1.xy = (
            d1[index1],
            brake1[index1]
        )

        brake_label1.set_text(
            f"D1  {brake1[index1]:.0f} Pa"
        )

        brake_label2.xy = (
            d2[index2],
            brake2[index2]
        )

        brake_label2.set_text(
            f"D2  {brake2[index2]:.0f} Pa"
        )

        # --------------------------------------------------
        # THROTTLE
        # --------------------------------------------------

        throttle_label1.xy = (
            d1[index1],
            throttle1[index1]
        )

        throttle_label1.set_text(
            f"D1  {throttle1[index1]:.1f}%"
        )

        throttle_label2.xy = (
            d2[index2],
            throttle2[index2]
        )

        throttle_label2.set_text(
            f"D2  {throttle2[index2]:.1f}%"
        )

        # --------------------------------------------------
        # LONGITUDINAL ACCELERATION
        # --------------------------------------------------

        accel_label1.xy = (
            d1[index1],
            lon_acc1[index1]
        )

        accel_label1.set_text(
            f"D1  {lon_acc1[index1]:+.2f}"
        )

        accel_label2.xy = (
            d2[index2],
            lon_acc2[index2]
        )

        accel_label2.set_text(
            f"D2  {lon_acc2[index2]:+.2f}"
        )

        # --------------------------------------------------
        # TIME DIFFERENCE
        # --------------------------------------------------

        delta_label.xy = (
            d1[index1],
            delta_time[index1]
        )

        delta_label.set_text(
            f"Δt {delta_time[index1]:+.3f}s"
        )

        # --------------------------------------------------
        # LATERAL G
        # --------------------------------------------------

        lat_label1.xy = (
            d1[index1],
            lat_acc1_g[index1]
        )

        lat_label1.set_text(
            f"D1  {lat_acc1_g[index1]:+.2f}g"
        )

        lat_label2.xy = (
            d2[index2],
            lat_acc2_g[index2]
        )

        lat_label2.set_text(
            f"D2  {lat_acc2_g[index2]:+.2f}g"
        )

        # --------------------------------------------------
        # G-G MARKERS
        # --------------------------------------------------

        gg_marker1.set_data(
            [lat_acc1_g[index1]],
            [lon_acc1_g[index1]]
        )

        gg_marker2.set_data(
            [lat_acc2_g[index2]],
            [lon_acc2_g[index2]]
        )

        # --------------------------------------------------
        # G-G TRAILS
        # --------------------------------------------------

        update_gg_trail(
            gg_trail1,
            lat_acc1_g,
            lon_acc1_g,
            index1,
            D1_COLOUR
        )

        update_gg_trail(
            gg_trail2,
            lat_acc2_g,
            lon_acc2_g,
            index2,
            D2_COLOUR
        )

        # --------------------------------------------------
        # REDRAW
        # --------------------------------------------------

        left_mpl.draw_idle()

        right_mpl.draw_idle()

    # ======================================================
    # G-G TRAIL UPDATE
    # ======================================================

    def update_gg_trail(
        collection,
        lat_values,
        lon_values,
        index,
        colour
    ):

        trail_length = 100

        start = max(
            0,
            index - trail_length
        )

        lat_trail = lat_values[
            start:index + 1
        ]

        lon_trail = lon_values[
            start:index + 1
        ]

        if len(lat_trail) < 2:

            collection.set_segments([])

            return

        points = np.column_stack([
            lat_trail,
            lon_trail
        ])

        segments = np.stack(
            [
                points[:-1],
                points[1:]
            ],
            axis=1
        )

        rgba = np.tile(
            to_rgba(colour),
            (
                len(segments),
                1
            )
        )

        rgba[:, 3] = np.linspace(
            0.02,
            0.80,
            len(segments)
        )

        collection.set_segments(
            segments
        )

        collection.set_color(
            rgba
        )

    # ======================================================
    # TRIANGLE MOVEMENT
    # ======================================================

    def move_triangle(
        triangle,
        x,
        y,
        index
    ):

        if len(x) < 2:
            return

        if index == 0:

            dx = x[1] - x[0]
            dy = y[1] - y[0]

        elif index == len(x) - 1:

            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]

        else:

            dx = (
                x[index + 1]
                - x[index - 1]
            )

            dy = (
                y[index + 1]
                - y[index - 1]
            )

        length = np.hypot(
            dx,
            dy
        )

        if length == 0:
            return

        dx /= length
        dy /= length

        nx = -dy
        ny = dx

        offset = 3.0

        centre_x = (
            x[index]
            + nx * offset
        )

        centre_y = (
            y[index]
            + ny * offset
        )

        triangle_length = 3.0
        triangle_width = 1.5

        tip_x = (
            centre_x
            - nx * triangle_length
        )

        tip_y = (
            centre_y
            - ny * triangle_length
        )

        base_x = (
            centre_x
            + nx * triangle_length
        )

        base_y = (
            centre_y
            + ny * triangle_length
        )

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

    # ======================================================
    # LEFT MOUSE CONTROLS
    # ======================================================

    left_dragging = False

    def left_press(event):

        nonlocal left_dragging

        if event.inaxes == ax1:

            left_dragging = True

            index1 = find_nearest(
                x1,
                y1,
                event.xdata,
                event.ydata
            )

            update_position(
                index1
            )

        elif event.inaxes == ax2:

            left_dragging = True

            index2 = find_nearest(
                x2,
                y2,
                event.xdata,
                event.ydata
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

        elif event.inaxes == ax_gg1:

            left_dragging = True

            distances = (
                (lat_acc1_g - event.xdata) ** 2
                + (lon_acc1_g - event.ydata) ** 2
            )

            index1 = np.argmin(
                distances
            )

            update_position(
                index1
            )

        elif event.inaxes == ax_gg2:

            left_dragging = True

            distances = (
                (lat_acc2_g - event.xdata) ** 2
                + (lon_acc2_g - event.ydata) ** 2
            )

            index2 = np.argmin(
                distances
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

        elif event.inaxes == ax_lat_speed1:

            left_dragging = True

            distances = (
                (s1 - event.xdata) ** 2
                + (lat_acc1_g - event.ydata) ** 2
            )

            index1 = np.argmin(
                distances
            )

            update_position(
                index1
            )

        elif event.inaxes == ax_lat_speed2:

            left_dragging = True

            distances = (
                (s2 - event.xdata) ** 2
                + (lat_acc2_g - event.ydata) ** 2
            )

            index2 = np.argmin(
                distances
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

    def left_motion(event):

        if not left_dragging:
            return

        if event.inaxes == ax1:

            index1 = find_nearest(
                x1,
                y1,
                event.xdata,
                event.ydata
            )

            update_position(
                index1
            )

        elif event.inaxes == ax2:

            index2 = find_nearest(
                x2,
                y2,
                event.xdata,
                event.ydata
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

        elif event.inaxes == ax_gg1:

            distances = (
                (lat_acc1_g - event.xdata) ** 2
                + (lon_acc1_g - event.ydata) ** 2
            )

            index1 = np.argmin(
                distances
            )

            update_position(
                index1
            )

        elif event.inaxes == ax_gg2:

            distances = (
                (lat_acc2_g - event.xdata) ** 2
                + (lon_acc2_g - event.ydata) ** 2
            )

            index2 = np.argmin(
                distances
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

        elif event.inaxes == ax_lat_speed1:

            distances = (
                (s1 - event.xdata) ** 2
                + (lat_acc1_g - event.ydata) ** 2
            )

            index1 = np.argmin(
                distances
            )

            update_position(
                index1
            )

        elif event.inaxes == ax_lat_speed2:

            distances = (
                (s2 - event.xdata) ** 2
                + (lat_acc2_g - event.ydata) ** 2
            )

            index2 = np.argmin(
                distances
            )

            index1 = match1_for_d2[
                index2
            ]

            update_position(
                index1
            )

    def left_release(event):

        nonlocal left_dragging

        left_dragging = False

    left_mpl.mpl_connect(
        "button_press_event",
        left_press
    )

    left_mpl.mpl_connect(
        "motion_notify_event",
        left_motion
    )

    left_mpl.mpl_connect(
        "button_release_event",
        left_release
    )

    # ======================================================
    # RIGHT MOUSE CONTROLS
    # ======================================================

    right_dragging = False

    distance_axes = [
        ax_speed,
        ax_brake,
        ax_throttle,
        ax_long_distance,
        ax_delta,
        ax_lat_distance
    ]

    def right_press(event):

        nonlocal right_dragging

        if event.inaxes not in distance_axes:
            return

        if event.xdata is None:
            return

        right_dragging = True

        index1 = np.argmin(
            np.abs(
                d1 - event.xdata
            )
        )

        update_position(
            index1
        )

    def right_motion(event):

        if not right_dragging:
            return

        if event.inaxes not in distance_axes:
            return

        if event.xdata is None:
            return

        index1 = np.argmin(
            np.abs(
                d1 - event.xdata
            )
        )

        update_position(
            index1
        )

    def right_release(event):

        nonlocal right_dragging

        right_dragging = False

    right_mpl.mpl_connect(
        "button_press_event",
        right_press
    )

    right_mpl.mpl_connect(
        "motion_notify_event",
        right_motion
    )

    right_mpl.mpl_connect(
        "button_release_event",
        right_release
    )

    # ======================================================
    # INITIAL POSITION
    # ======================================================

    update_position(
        0
    )

    root.update_idletasks()

    right_canvas.configure(
        scrollregion=right_canvas.bbox("all")
    )

    # ======================================================
    # CLOSE CLEANLY
    # ======================================================

    def close_window():

        plt.close(
            left_fig
        )

        plt.close(
            right_fig
        )

        root.destroy()

    root.protocol(
        "WM_DELETE_WINDOW",
        close_window
    )

    root.mainloop()

