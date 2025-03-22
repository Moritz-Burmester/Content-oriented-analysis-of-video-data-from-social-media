import pandas as pd

"""
This scripts sorts the file filled from video_hashes.py and looks for duplicates and saves them in two seperate files.
The one file contains only the duplicatse and the other the original and the duplicates
Requirement: Csv File with id,hash1,hash2
"""

video_hash_values_db = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/video_hash_values.csv"
duplicates_db = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv"
duplicates_to_originals_db = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates_to_originals.csv"

df = pd.read_csv(video_hash_values_db)

# Sort the data after the dates found in id. This is important to find the first video of a duplicate stack
"""
df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
df = df.sort_values(by="date")
df.drop(columns=["date"]).to_csv(video_hash_values_db, index = False)
"""

# To keep the first found video as the original
df.reset_index(inplace=True)
df.sort_values(by="index", inplace=True)

duplicates_dict = []
all_duplicates = []

#Group by hash-values
for _, group in df.groupby(["hash1", "hash2"]):
    if len(group) > 1:
        original = group.iloc[0]
        original_id = original["id"]
        hash1, hash2 = original["hash1"], original["hash2"]

        #iterate trough the duplicates and stores them
        for _, duplicate in group.iloc[1:].iterrows():
            duplicates_dict.append({
                'original_id': original_id,
                'duplicate_id': duplicate['id'],
                'hash1': hash1,
                'hash2': hash2
            })
            all_duplicates.append(duplicate['id'])

# Save grouped duplicates in the correct location
grouped_df = pd.DataFrame(duplicates_dict)
grouped_df.to_csv(duplicates_to_originals_db, index=False)

# Save all duplicates in a separate list
duplicates_df = pd.DataFrame({'duplicate_id': all_duplicates})
duplicates_df.to_csv(duplicates_db, index=False)