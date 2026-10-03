from src.data_processing import load_data, assign_laps, assign_distance
from src.lap_analysis import calculate_lap_times
from src.plotting import plot_lap_times, plot_speed_map, interactive_track, compare_drivers

def main():
    driver1, driver2, laps = load_data()

    assign_laps(driver1, driver2, laps)

    calculate_lap_times(laps)

    plot_lap_times(laps)

    driver1 = assign_distance(driver1)
    driver2 = assign_distance(driver2)

    # interactive_track(driver1, 2)

    # plot_speed_map(driver1, 3)



    compare_drivers(driver1, driver2, 9)

main()



