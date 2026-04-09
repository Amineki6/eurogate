import pandas as pd
import matplotlib.pyplot as plt
from data_prep.preprocess_training_data import prepare_feature_frame, TIME_COL

def main():
    print("Loading and preprocessing data...")
    df = prepare_feature_frame("reefer_release.csv")
    
    print("Plotting...")
    plt.figure(figsize=(12, 6))
    plt.plot(df[TIME_COL], df["container_count"], label="Container Count (Occupancy)")
    plt.xlabel("Time")
    plt.ylabel("Occupancy")
    plt.title("Terminal Occupancy per Hour")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("occupancy.png")
    print("Plot saved to occupancy.png")

if __name__ == "__main__":
    main()
