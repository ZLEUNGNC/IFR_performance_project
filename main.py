from src.data_processing import load_data, assign_laps
from src.lap_analysis import calculate_lap_times
from src.plotting import plot_lap_times, plot_speed_map


def main():
    driver1, driver2, laps = load_data()

    assign_laps(driver1, driver2, laps)

    calculate_lap_times(laps)

    plot_lap_times(laps)

    lap_number = int(input("Enter lap number: "))


    driverpd = input("Enter \"driver1\" or \"driver2\": ") #input driver1 or driver2

    if user_input == "driver1":
        driverpd = driver1
    elif user_input == "driver2":
        driverpd = driver2
    else:
        driverpd = None
    print("Warning: Invalid input. driverpd was not set to a DataFrame.")

    plot_speed_map(driverpd, lap_number)


if __name__ == "__main__":
    main()