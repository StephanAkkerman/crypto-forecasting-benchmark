import os

# Local Import
import config

if __name__ == "__main__":
    # Start by testing if the data is available
    for coin in config.all_coins:
        if os.path.exists(f"{config.coin_dir}/{coin}"):
            # Test if the .csv files exist in the folder
            for time_frame in config.time_frames:
                if not os.path.exists(
                    f"{config.coin_dir}/{coin}/{coin}USDT_{time_frame}.csv"
                ):
                    print(
                        f"Data for {coin}USDT_{time_frame}.csv is missing. Please download it first."
                    )
                    continue
        else:
            print(f"Data for {coin} is not available. Please download it first.")
            continue

    # Run the analysis
