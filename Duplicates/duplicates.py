import pandas as pd

video_hash_values_db = "/home/mburmest/bachelorarbeit/Duplicates/video_hash_values.csv"
duplicates_db = "/home/mburmest/bachelorarbeit/Duplicates/duplicates.csv"
duplicates_to_originals_db = "/home/mburmest/bachelorarbeit/Duplicates/duplicates_to_originals.csv"

# Sort the data after the dates found in id. This is important to find the first video of a duplicate stack
df = pd.read_csv(video_hash_values_db)
df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
