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

    print(
        f"D1 brake column: {brake_col1}"
    )

    print(
        f"D2 brake column: {brake_col2}"
    )

    print(
        f"D1 throttle column: {throttle_col1}"
    )

    print(
        f"D2 throttle column: {throttle_col2}"
    )

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

        if np.nanmax(np.abs(series)) <= 1.05:
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

    lat_acc1_g = lap1[LAT_ACC_COL].to_numpy()
    lat_acc2_g = lap2[LAT_ACC_COL].to_numpy()

    lon_acc1_g = lap1[LON_ACC_COL].to_numpy()
    lon_acc2_g = lap2[LON_ACC_COL].to_numpy()

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

    def find_nearest(x, y, mouse_x, mouse_y):

        distances = (
            (x - mouse_x) ** 2
            + (y - mouse_y) ** 2
        )

        return np.argmin(distances)

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
        for x, y in zip(x1, y1)
    ])

    match1_for_d2 = np.array([
        find_nearest(
            x1,
            y1,
            x,
            y
        )
        for x, y in zip(x2, y2)
    ])

    # ======================================================
    # D2 - D1 TIME DIFFERENCE
    # ======================================================

    delta_time = (
        t2_rel[match2_for_d1]
        - t1_rel
    )

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

    root.grid_rowconfigure(
        0,
        weight=1
    )

    root.grid_columnconfigure(
        0,
        weight=1
    )

    root.grid_columnconfigure(
        1,
        weight=2
    )

    # ======================================================
    # LEFT FIXED PANEL
    # ======================================================

    left_frame = tk.Frame(root)

    left_frame.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    # ======================================================
    # RIGHT SCROLLABLE PANEL
    # ======================================================

    right_frame = tk.Frame(root)

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
        highlightthickness=0
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
        right_canvas
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
    # MOUSE WHEEL FOR RIGHT PANEL
    # ======================================================

    def scroll_right(event):

        if hasattr(event, "delta") and event.delta != 0:

            right_canvas.yview_scroll(
                int(-event.delta / 120),
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
    # ======================================================

    left_fig = plt.Figure(
        figsize=(6, 10),
        dpi=100
    )

    left_gs = left_fig.add_gridspec(
        4,
        3,
        height_ratios=[
            1.05,
            1.05,
            0.10,
            0.90
        ],
        width_ratios=[
            1,
            1,
            0.07
        ],
        hspace=0.55,
        wspace=0.30
    )

    ax1 = left_fig.add_subplot(
        left_gs[0, 0:2]
    )

    ax2 = left_fig.add_subplot(
        left_gs[1, 0:2]
    )

    cax = left_fig.add_subplot(
        left_gs[0:2, 2]
    )

    ax_gg = left_fig.add_subplot(
        left_gs[3, 0]
    )

    ax_lat_speed = left_fig.add_subplot(
        left_gs[3, 1]
    )

    # ======================================================
    # RIGHT FIGURE
    # ======================================================

    right_fig = plt.Figure(
        figsize=(11, 24),
        dpi=100
    )

    right_gs = right_fig.add_gridspec(
        4,
        1,
        height_ratios=[
            1.0,
            1.15,
            1.0,
            1.0
        ],
        hspace=0.55
    )

    ax_speed = right_fig.add_subplot(
        right_gs[0, 0]
    )

    ax_control = right_fig.add_subplot(
        right_gs[1, 0]
    )

    ax_delta = right_fig.add_subplot(
        right_gs[2, 0]
    )

    ax_lat_distance = right_fig.add_subplot(
        right_gs[3, 0]
    )

    # ======================================================
    # EMBED FIGURES
    # ======================================================

    left_mpl = FigureCanvasTkAgg(
        left_fig,
        master=left_frame
    )

    left_mpl.draw()

    left_mpl_widget = left_mpl.get_tk_widget()

    left_mpl_widget.pack(
        fill="both",
        expand=True
    )

    right_mpl = FigureCanvasTkAgg(
        right_fig,
        master=right_content
    )

    right_mpl.draw()

    right_mpl_widget = right_mpl.get_tk_widget()

    right_mpl_widget.pack(
        fill="both",
        expand=True
    )

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
        s=6
    )

    ax2.scatter(
        x2,
        y2,
        c=s2,
        cmap="turbo",
        norm=speed_norm,
        s=6
    )

    left_fig.colorbar(
        scatter1,
        cax=cax,
        label="Speed (km/h)"
    )

    # ======================================================
    # TRACK FORMATTING
    # ======================================================

    ax1.set_title(
        f"Driver 1 - Lap {lap_number}"
    )

    ax2.set_title(
        f"Driver 2 - Lap {lap_number}"
    )

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

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

    ax1.add_patch(triangle1)
    ax2.add_patch(triangle2)

    # ======================================================
    # TRACK INFORMATION
    # ======================================================

    text1 = ax1.text(
        0.02,
        0.97,
        "",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8
        )
    )

    text2 = ax2.text(
        0.02,
        0.97,
        "",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8
        )
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
    # SPEED ARROWS
    # ======================================================

    speed_arrow1 = ax_speed.annotate(
        "",
        xy=(d1[0], s1[0]),
        xycoords="data",
        xytext=(0.98, 0.98),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D1_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D1_COLOUR,
            alpha=0.9
        )
    )

    speed_arrow2 = ax_speed.annotate(
        "",
        xy=(d2[0], s2[0]),
        xycoords="data",
        xytext=(0.98, 0.74),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D2_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D2_COLOUR,
            alpha=0.9
        )
    )

    # ======================================================
    # BRAKE / THROTTLE / LONG ACCEL
    # ======================================================

    brake_line1, = ax_control.plot(
        d1,
        brake1,
        color=D1_COLOUR,
        linestyle="-",
        label="D1 Brake"
    )

    brake_line2, = ax_control.plot(
        d2,
        brake2,
        color=D2_COLOUR,
        linestyle="-",
        label="D2 Brake"
    )

    ax_control.set_ylabel(
        "Brake pressure (Pa)"
    )

    ax_control.set_xlabel(
        "Distance (m)"
    )

    ax_control.grid(
        True,
        alpha=0.3
    )

    # Throttle - right axis
    ax_throttle = ax_control.twinx()

    throttle_line1, = ax_throttle.plot(
        d1,
        throttle1,
        color=D1_COLOUR,
        linestyle="--",
        label="D1 Throttle"
    )

    throttle_line2, = ax_throttle.plot(
        d2,
        throttle2,
        color=D2_COLOUR,
        linestyle="--",
        label="D2 Throttle"
    )

    ax_throttle.set_ylabel(
        "Throttle (%)"
    )

    ax_throttle.set_ylim(
        0,
        100
    )

    # Longitudinal acceleration - second left axis
    ax_accel = ax_control.twinx()

    ax_accel.spines["right"].set_visible(False)

    ax_accel.spines["left"].set_position(
        ("axes", -0.10)
    )

    ax_accel.spines["left"].set_visible(True)

    ax_accel.yaxis.set_label_position(
        "left"
    )

    ax_accel.yaxis.set_ticks_position(
        "left"
    )

    accel_line1, = ax_accel.plot(
        d1,
        lon_acc1,
        color=D1_COLOUR,
        linestyle=":",
        label="D1 Long Accel"
    )

    accel_line2, = ax_accel.plot(
        d2,
        lon_acc2,
        color=D2_COLOUR,
        linestyle=":",
        label="D2 Long Accel"
    )

    ax_accel.set_ylabel(
        "Longitudinal acceleration (m/s²)"
    )

    ax_control.set_title(
        "Brake Pressure / Throttle / Longitudinal Acceleration"
    )

    # Combined legend
    control_handles = [
        Line2D(
            [0],
            [0],
            color=D1_COLOUR,
            linestyle="-",
            label="Driver 1 Brake"
        ),
        Line2D(
            [0],
            [0],
            color=D2_COLOUR,
            linestyle="-",
            label="Driver 2 Brake"
        ),
        Line2D(
            [0],
            [0],
            color=D1_COLOUR,
            linestyle="--",
            label="Driver 1 Throttle"
        ),
        Line2D(
            [0],
            [0],
            color=D2_COLOUR,
            linestyle="--",
            label="Driver 2 Throttle"
        ),
        Line2D(
            [0],
            [0],
            color=D1_COLOUR,
            linestyle=":",
            label="Driver 1 Long Accel"
        ),
        Line2D(
            [0],
            [0],
            color=D2_COLOUR,
            linestyle=":",
            label="Driver 2 Long Accel"
        )
    ]

    ax_control.legend(
        handles=control_handles,
        fontsize=8,
        ncol=2,
        loc="upper left"
    )

    # Control arrows
    control_arrow1 = ax_control.annotate(
        "",
        xy=(d1[0], brake1[0]),
        xycoords="data",
        xytext=(0.98, 0.98),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D1_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D1_COLOUR,
            alpha=0.9
        )
    )

    control_arrow2 = ax_control.annotate(
        "",
        xy=(d2[0], brake2[0]),
        xycoords="data",
        xytext=(0.98, 0.69),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D2_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D2_COLOUR,
            alpha=0.9
        )
    )

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
        "D2 - D1 Time Difference"
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

    delta_arrow = ax_delta.annotate(
        "",
        xy=(d1[0], delta_time[0]),
        xycoords="data",
        xytext=(0.98, 0.98),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
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

    lat_distance_arrow1 = ax_lat_distance.annotate(
        "",
        xy=(d1[0], lat_acc1_g[0]),
        xycoords="data",
        xytext=(0.98, 0.98),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D1_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D1_COLOUR,
            alpha=0.9
        )
    )

    lat_distance_arrow2 = ax_lat_distance.annotate(
        "",
        xy=(d2[0], lat_acc2_g[0]),
        xycoords="data",
        xytext=(0.98, 0.72),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=D2_COLOUR,
            linewidth=1.5
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=D2_COLOUR,
            alpha=0.9
        )
    )

    # ======================================================
    # G-G DIAGRAM
    # ======================================================

    ax_gg.set_title(
        "G-G Diagram"
    )

    ax_gg.set_xlabel(
        "Lateral acceleration (g)"
    )

    ax_gg.set_ylabel(
        "Longitudinal acceleration (g)"
    )

    ax_gg.grid(
        True,
        alpha=0.2
    )

    ax_gg.axhline(
        0,
        color="black",
        linewidth=0.7,
        alpha=0.4
    )

    ax_gg.axvline(
        0,
        color="black",
        linewidth=0.7,
        alpha=0.4
    )

    # Full faint traces
    ax_gg.plot(
        lat_acc1_g,
        lon_acc1_g,
        color=D1_COLOUR,
        alpha=0.10,
        linewidth=1
    )

    ax_gg.plot(
        lat_acc2_g,
        lon_acc2_g,
        color=D2_COLOUR,
        alpha=0.10,
        linewidth=1
    )

    ax_gg.set_aspect(
        "equal",
        adjustable="box"
    )

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

    if not np.isfinite(gg_max) or gg_max == 0:
        gg_max = 1.0

    gg_max *= 1.10

    ax_gg.set_xlim(
        -gg_max,
        gg_max
    )

    ax_gg.set_ylim(
        -gg_max,
        gg_max
    )

    # Decaying trails
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

    ax_gg.add_collection(
        gg_trail1
    )

    ax_gg.add_collection(
        gg_trail2
    )

    # Small current-position dots
    gg_marker1, = ax_gg.plot(
        [lat_acc1_g[0]],
        [lon_acc1_g[0]],
        marker="o",
        markersize=4,
        color=D1_COLOUR,
        linestyle="",
        zorder=10
    )

    gg_marker2, = ax_gg.plot(
        [lat_acc2_g[0]],
        [lon_acc2_g[0]],
        marker="o",
        markersize=4,
        color=D2_COLOUR,
        linestyle="",
        zorder=10
    )

    ax_gg.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=D1_COLOUR,
                label="Driver 1"
            ),
            Line2D(
                [0],
                [0],
                color=D2_COLOUR,
                label="Driver 2"
            )
        ],
        fontsize=8,
        loc="lower right"
    )

    # ======================================================
    # LATERAL G VS SPEED
    # STATIC SCATTER
    # ======================================================

    ax_lat_speed.scatter(
        s1,
        lat_acc1_g,
        s=5,
        color=D1_COLOUR,
        alpha=0.20,
        label="Driver 1"
    )

    ax_lat_speed.scatter(
        s2,
        lat_acc2_g,
        s=5,
        color=D2_COLOUR,
        alpha=0.20,
        label="Driver 2"
    )

    ax_lat_speed.axhline(
        0,
        color="black",
        linewidth=0.7,
        alpha=0.4
    )

    ax_lat_speed.set_title(
        "Lateral G vs Speed"
    )

    ax_lat_speed.set_xlabel(
        "Speed (km/h)"
    )

    ax_lat_speed.set_ylabel(
        "Lateral G"
    )

    ax_lat_speed.grid(
        True,
        alpha=0.2
    )

    ax_lat_speed.legend(
        fontsize=8
    )

    # ======================================================
    # MASTER UPDATE
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

        # Old sections fade; newest section is strongest
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

    def update_position(index1):

        # --------------------------------------------------
        # Match physical D2 position
        # --------------------------------------------------

        index2 = match2_for_d1[index1]

        # --------------------------------------------------
        # Tracks
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
        # Track text
        # --------------------------------------------------

        text1.set_text(
            f"D1\n"
            f"Distance: {d1[index1]:.2f} m\n"
            f"Speed: {s1[index1]:.1f} km/h"
        )

        text2.set_text(
            f"D2\n"
            f"Distance: {d2[index2]:.2f} m\n"
            f"Speed: {s2[index2]:.1f} km/h"
        )

        # --------------------------------------------------
        # Speed graph
        # --------------------------------------------------

        speed_arrow1.xy = (
            d1[index1],
            s1[index1]
        )

        speed_arrow1.set_text(
            f"D1\n"
            f"Distance: {d1[index1]:.2f} m\n"
            f"Speed: {s1[index1]:.1f} km/h"
        )

        speed_arrow2.xy = (
            d2[index2],
            s2[index2]
        )

        speed_arrow2.set_text(
            f"D2\n"
            f"Distance: {d2[index2]:.2f} m\n"
            f"Speed: {s2[index2]:.1f} km/h"
        )

        # --------------------------------------------------
        # Control graph
        # --------------------------------------------------

        control_arrow1.xy = (
            d1[index1],
            brake1[index1]
        )

        control_arrow1.set_text(
            f"D1\n"
            f"Distance: {d1[index1]:.2f} m\n"
            f"Brake: {brake1[index1]:.0f} Pa\n"
            f"Throttle: {throttle1[index1]:.1f} %\n"
            f"Long Accel: {lon_acc1[index1]:.2f} m/s²"
        )

        control_arrow2.xy = (
            d2[index2],
            brake2[index2]
        )

        control_arrow2.set_text(
            f"D2\n"
            f"Distance: {d2[index2]:.2f} m\n"
            f"Brake: {brake2[index2]:.0f} Pa\n"
            f"Throttle: {throttle2[index2]:.1f} %\n"
            f"Long Accel: {lon_acc2[index2]:.2f} m/s²"
        )

        # --------------------------------------------------
        # Time difference
        # --------------------------------------------------

        delta_arrow.xy = (
            d1[index1],
            delta_time[index1]
        )

        delta_arrow.set_text(
            f"D2 - D1\n"
            f"Distance: {d1[index1]:.2f} m\n"
            f"Δt: {delta_time[index1]:+.3f} s"
        )

        # --------------------------------------------------
        # Lateral G distance
        # --------------------------------------------------

        lat_distance_arrow1.xy = (
            d1[index1],
            lat_acc1_g[index1]
        )

        lat_distance_arrow1.set_text(
            f"D1\n"
            f"Distance: {d1[index1]:.2f} m\n"
            f"Lat G: {lat_acc1_g[index1]:+.2f} g"
        )

        lat_distance_arrow2.xy = (
            d2[index2],
            lat_acc2_g[index2]
        )

        lat_distance_arrow2.set_text(
            f"D2\n"
            f"Distance: {d2[index2]:.2f} m\n"
            f"Lat G: {lat_acc2_g[index2]:+.2f} g"
        )

        # --------------------------------------------------
        # G-G
        # --------------------------------------------------

        gg_marker1.set_data(
            [lat_acc1_g[index1]],
            [lon_acc1_g[index1]]
        )

        gg_marker2.set_data(
            [lat_acc2_g[index2]],
            [lon_acc2_g[index2]]
        )

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
        # Redraw
        # --------------------------------------------------

        left_mpl.draw_idle()
        right_mpl.draw_idle()

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

            dx = x[index + 1] - x[index - 1]
            dy = y[index + 1] - y[index - 1]

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

            index1 = match1_for_d2[index2]

            update_position(
                index1
            )

        elif event.inaxes == ax_gg:

            left_dragging = True

            distance1 = (
                (lat_acc1_g - event.xdata) ** 2
                + (lon_acc1_g - event.ydata) ** 2
            )

            distance2 = (
                (lat_acc2_g - event.xdata) ** 2
                + (lon_acc2_g - event.ydata) ** 2
            )

            if distance1.min() <= distance2.min():

                index1 = np.argmin(
                    distance1
                )

            else:

                index2 = np.argmin(
                    distance2
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

        elif event.inaxes == ax_gg:

            distance1 = (
                (lat_acc1_g - event.xdata) ** 2
                + (lon_acc1_g - event.ydata) ** 2
            )

            distance2 = (
                (lat_acc2_g - event.xdata) ** 2
                + (lon_acc2_g - event.ydata) ** 2
            )

            if distance1.min() <= distance2.min():

                index1 = np.argmin(
                    distance1
                )

            else:

                index2 = np.argmin(
                    distance2
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
        ax_control,
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
    # START
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

