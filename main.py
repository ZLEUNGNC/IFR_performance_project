from src.data_processing import load_data, assign_laps
from src.lap_analysis import calculate_lap_times
from src.plotting import plot_lap_times, plot_speed_map


def main():
    driver1, driver2, laps = load_data()

    assign_laps(driver1, driver2, laps)

    calculate_lap_times(laps)

    plot_lap_times(laps)

    lap_number = int(input("Enter lap number: "))

    plot_speed_map(driver1, lap_number)


if __name__ == "__main__":
    main()