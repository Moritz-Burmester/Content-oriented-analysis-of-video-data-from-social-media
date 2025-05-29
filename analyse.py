import datetime
import glob
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

"""
This file contains helper functions for analyzing and visualizing the solutions:
- Collecting exceptions
- Printing solution-related information
- Generating graphs
- Formatting and normalizing the results
"""

# Output Path for the Diagrams
output_path = "/work/mburmest/bachelorarbeit/Diagrams/"

# Paths to all generated solutions
solution_clip = "/work/mburmest/bachelorarbeit/Solution/clip_solution.csv"
solution_videollava_1 = "/work/mburmest/bachelorarbeit/Solution/videollava_solution_1.csv"
solution_videollava_2 = "/work/mburmest/bachelorarbeit/Solution/videollava_solution_2.csv"
solutions_videollava_3 = "/work/mburmest/bachelorarbeit/Solution/videollava_solution_3.csv"
solutions_videollava_1000 = "/work/mburmest/bachelorarbeit/Solution/videollava_solution_1000.csv"
solution_videochatgpt_1 = "/work/mburmest/bachelorarbeit/Solution/videochatgpt_solution_1.csv"
solution_videochatgpt_2 = "/work/mburmest/bachelorarbeit/Solution/videochatgpt_solution_2.csv"
solution_videochatgpt_3 = "/work/mburmest/bachelorarbeit/Solution/videochatgpt_solution_3.csv"
solution_pandagpt_1 = "/work/mburmest/bachelorarbeit/Solution/pandagpt_solution_1.csv"
solution_pandagpt_2 = "/work/mburmest/bachelorarbeit/Solution/pandagpt_solution_2.csv"
solution_pandagpt_3 = "/work/mburmest/bachelorarbeit/Solution/pandagpt_solution_3.csv"
solution_pandagpt_1000 = "/work/mburmest/bachelorarbeit/Solution/pandagpt_solution_1000.csv"

# Collection of Solutions without the subsets
solutions = [solution_clip, 
             solution_videollava_1, solution_videollava_2, solutions_videollava_3, 
             solution_videochatgpt_1, solution_videochatgpt_2, solution_videochatgpt_3, 
             solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3]

# Paths to all exceptions
exception_videollava = "/work/mburmest/bachelorarbeit/Exceptions/videollava_exception.csv"
exception_videochatgpt = "/work/mburmest/bachelorarbeit/Exceptions/videochatgpt_exception.csv"
exception_pandagpt = "/work/mburmest/bachelorarbeit/Exceptions/pandagpt_exception.csv"

# Collection of Exceptions
exceptions_files = [exception_videollava, exception_videochatgpt, exception_pandagpt]


def exceptions():
    """
    Takes in all exceptions declared in exception_files and returns a list of IDs.
    Prints out the number of exceptions in each file and in total.

    Returns:
        list: A list of video ids.
    """

    exception_ids = set()
    for exception in exceptions_files:
        with open(exception) as f:
            ids = re.findall(r"id_\d+_\d{4}-\d{2}-\d{2}", f.read())
            print(f"{exception}: {len(ids)} IDs")

    for exception in exceptions_files:
        with open(exception) as f:
            exception_ids.update(re.findall(r"id_\d+_\d{4}-\d{2}-\d{2}", f.read()))

    print(f"\nTotal exception_ids: {len(exception_ids)}")
    return exception_ids

def info_subset():
    """
    Gets all the videos from the dataset, removes the duplicates and the videos per year.
    During this process information about the dataset and subset is printed.
    """

    videos = glob.glob("/ceph/lprasse/ClimateVisions/Videos/*/*/*.mp4")
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv")
    duplicates = df_dups["id"].astype(str)

    print(f"Videos total: {len(videos)}")

    # Create a DataFrame from the video paths
    df_videos = pd.DataFrame({"path": videos})

    # Extract ID and date from filenames using regex
    # Example filename: id_1228086819004321798_2020-02-14.mp4
    df_videos["filename"] = df_videos["path"].str.extract(r"([^/]+\.mp4)$")
    df_videos["id"] = df_videos["filename"].str.extract(r"(id_\d+_\d{4}-\d{2}-\d{2})")
    df_videos["date"] = df_videos["id"].str.extract(r"(\d{4}-\d{2}-\d{2})")
    df_videos["year"] = pd.to_datetime(df_videos["date"], errors="coerce").dt.year

    # Remove duplicates
    df_filtered = df_videos[~df_videos["id"].isin(duplicates)]

    print(f"Videos after duplicates: {len(df_filtered)}")

    # Count videos per year (for 2019–2022 only)
    for year in [2019, 2020, 2021, 2022]:
        count = df_filtered[df_filtered["year"] == year].shape[0]
        print(f"Videos in {year}: {count}")

def same_subset(paths):
    """
    Reads a list of paths and creates a list of intersecting video ids.
    Each file is then overwritten to only keep those ids.
    
    Parameters:
        paths (list): list of paths for files
    """

    common_ids = None

    for file in paths:
        df = pd.read_csv(file)
        current_ids = set(df["id"].unique())
        
        if common_ids is None:
            common_ids = current_ids
        else:
            common_ids &= current_ids  # Intersection update
            
    print(f"Final number of common IDs: {len(common_ids)}")


    for file in paths:
        # Read entire file
        df = pd.read_csv(file)
        
        # Filter to only keep rows with common IDs
        filtered_df = df[df["id"].isin(common_ids)]
        
        # Overwrite original file
        filtered_df.to_csv(file, index=False)
        print(f"Overwritten {file} - now contains {len(filtered_df)} rows")

def correct_spelling(path):
    """
    Uses the capitalize_words(text) method to format each word in the file so that it starts with a capitalized letter, followed by lowercase letters.
    This overwrites the information in the file.

    Parameters:
        path (str): Path to the file.
    """

    def capitalize_words(text):
        """
        Formats a string to start with a capitalized letter, followed by lowercase letters.
        
        Parameters:
            text (str): String to format
        """

        if pd.isna(text):  # Skip NaN values
            return text
        
        # Process each category separated by |
        categories = str(text).split("|")
        capitalized_categories = []

        for category in categories: # Capitalize each word in the part
            
            capitalized_category = " ".join(word.capitalize() for word in category.split())
            capitalized_categories.append(capitalized_category)

        # Rejoin with | separator
        return "|".join(capitalized_categories)

    df = pd.read_csv(path)
    df = df.map(capitalize_words)
    df["id"] = df["id"].str.lower()
    # Optional:
    # df = sort_ids(df) 

    df.to_csv(path, index=False)

def fix_animals_column(df):
    """
    Takes in a pandas dataframe and replaces the category "Other" with "Other Animals".
    This dataframe gets returned.
    
    Parameters:
        df (pandas dataframe): dataframe with "Other" instead of "Other Animals"

    Returns:
        Formatted pandas df
    """

    def replace_other(cell):
        """
        Takes a cell of a df and  replaces the category "Other" with "Other Animals"
        
        Parameters:
            cell (str)

        Returns:
            Formatted cell (str)
        """

        if pd.isna(cell):
            return cell
        parts = [p.strip() for p in str(cell).split("|")]
        parts = ["Other Animals" if p == "Other" else p for p in parts]
        return "|".join(parts)

    df["animals"] = df["animals"].apply(replace_other)
    return df

def merge_categories(df):
    """
    Takes in a pandas dataframe and merges two categories into one.
    The categories can be changed in the code.
    This dataframe gets returned.
    
    Parameters:
        df (pandas dataframe)

    Returns:
        Formatted pandas df
    """

    def merge(cell):
        """
        Takes a cell of a df and merges two categories.
        Here Forest and Jungle
        
        Parameters:
            cell (str)

        Returns:
            Formatted cell (str)
        """

        if pd.isna(cell):
            return cell
        parts = [p.strip() for p in str(cell).split("|")]
        if "Forest" not in parts and "Jungle" not in parts:
            return "|".join(parts)
        # Remove both if either is present, then add combined
        parts = [p for p in parts if p not in {"Forest", "Jungle"}]
        parts.append("Forest, Jungle")
        return "|".join(parts)

    df["setting"] = df["setting"].apply(merge)
    return df

def sort_ids(df):
    """    
    Takes in a pandas dataframe and sorts it according to the date and number found in the video id.

    Parameters:
        df (pandas dataframe)

    Returns:
        sorted pandas dataframe
    """

    df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
    df["numeric_id"] = df["id"].str.split("_").str[1].astype(int)
    return df.sort_values(by=["date", "numeric_id"]).drop(columns=["date", "numeric_id"])

def visualize(path):
    """
    Takes a solution file path and visualizies the content of each column in a bar chart.
    This bar-chart is saved under the output_path/single/modelname/solution_X_column.png

    Parameters:
        path (str): Path of solution file.
    """

    # Select columns
    PLOT_COLS = ["animals", "climateactions", "consequences", "setting", "type"]
    all_data = {col: defaultdict(dict) for col in PLOT_COLS}

    df = pd.read_csv(path)
    exceptions_ids = exceptions()
    df = df[~df["id"].isin(exceptions_ids)]
    name = path.split("/")[-1].split(".")[0]

    # Select Fontsize
    fontsize_global = 24

    for category in PLOT_COLS:
        
        split_categories = df[category].astype(str).str.split("|").explode().str.strip()
        split_categories = split_categories[(split_categories != "") & (split_categories != "nan")]

        category_counts = split_categories.value_counts().sort_values(ascending=False)
        category_percents = ((category_counts / len(df)) * 100).round(1)

        # Store for later multi-dataset plot
        for label in category_counts.index:
            all_data[category][label][name] = category_counts[label]

        # Print data table for confirmation
        print(f"\n{name} - {category} Analysis:")
        result_df = pd.DataFrame({
            "Category": category_counts.index,
            "Count": category_counts.values,
            "Percentage": category_percents.values
        })
        print(result_df.to_string(index=False))

        plt.rcParams.update({"font.size": fontsize_global})

        # Combined bar chart (only bars for count, percentage as text labels)
        fig, ax1 = plt.subplots(figsize=(18, 6))

        x = category_counts.index
        counts = category_counts.values
        percents = category_percents.values

        if path == solutions_videollava_1000 or path == solution_pandagpt_1000:
            bars = ax1.bar(x, counts, color="orange", label="Count")
        else:
            bars = ax1.bar(x, counts, color="teal", label="Count")

        ax1.set_ylabel("Count", fontsize=fontsize_global)
        # Optional title
        # ax1.set_title(f"{name} - {category} (Count & Percentage Labels)")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Rotate x-tick labels
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels(x, rotation=45, ha="right", fontsize=fontsize_global)

        # Annotate bars with count + percentage
        for i, (bar, pct) in enumerate(zip(bars, percents)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{pct}", ha="center", va="bottom", fontsize=fontsize_global - 1)

        #fig.tight_layout()
        plt.savefig(f"{output_path}/Single/{name}_{category}.png", bbox_inches="tight", dpi=300)
        plt.close()

def combined_bar_charts(csv_paths):
    """
    Takes in a list of path from solution files and creates a combined bar chart for the selected columns.
    This bar-chart is saved under the output_path/Combined/combined_analysis_column.png

    Parameters:
        csv_path (list): list of paths
    """

    # Select Columns, Fontsize and Bar Colors.
    PLOT_COLS = ["type"]
    fontsize_global = 24  
    custom_colors = ["teal", "orange", "purple", "green", "crimson"]  

    all_data = {col: defaultdict(dict) for col in PLOT_COLS}
    dfs = [pd.read_csv(path) for path in csv_paths]
    names = [path.split("/")[-1].split(".")[0].split("_")[0] for path in csv_paths]

    # Change names for legend
    for i ,name in enumerate(names):
        if name == "videochatgpt": names[i] = "Video-ChatGPT"
        if name == "pandagpt": names[i] = "PandaGPT"
        if name == "clip": names[i] = "CLIP"

    # Step 1: Exclude rows with "No Class Found" in any PLOT_COLS
    blacklisted_ids = set()
    for df in dfs:
        for col in PLOT_COLS:
            mask = df[col].astype(str).str.contains("No Class Found", na=False)            

            blacklisted_ids.update(df.loc[mask, "id"].tolist())

    dfs = [df[~df["id"].isin(blacklisted_ids)].copy() for df in dfs]

    plt.rcParams.update({"font.size": fontsize_global})

    for column in PLOT_COLS:
        category_counts_per_file = []
        all_categories = set()

        # Collect counts and all unique categories
        for df in dfs:
            col_data = df[column].dropna().astype(str).str.split("|").explode().str.strip()
            counts = col_data.value_counts()
            category_counts_per_file.append(counts)
            all_categories.update(counts.index)

        all_categories = sorted(all_categories)
        data_matrix = []

        # Build a matrix with counts per file per category
        for counts in category_counts_per_file:
            row = [counts.get(cat, 0) for cat in all_categories]
            data_matrix.append(row)

        # Transpose for plotting
        data = np.array(data_matrix)
        x = np.arange(len(all_categories))
        width = 0.9 / len(dfs)  # width per bar

        fig, ax = plt.subplots(figsize=(18, 7))
        for i, file_counts in enumerate(data):
            color = custom_colors[i % len(custom_colors)]  # Safeguard against overflow
            bar = ax.bar(x + i * width, file_counts, width, color=color, label=names[i])

            # Count + Percentage label — editable section
            total = sum(file_counts)
            for idx, height in enumerate(file_counts):
                percent = (height / total * 100) if total > 0 else 0
                if percent < 1:
                    string_percent = ""
                else:
                    string_percent = f"{percent:.1f}"

                ax.text(
                    x[idx] + i * width,
                    height,
                    string_percent,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize_global - 4,
                    rotation=0,
                )
    

        ax.set_ylabel("Count", fontsize=fontsize_global)
        ax.set_xticks(x + width * (len(dfs) - 1) / 2)
        ax.set_xticklabels(all_categories, rotation=45, ha="right", fontsize=fontsize_global)
        ax.legend(fontsize=fontsize_global)
        ax.grid(axis="y", linestyle="--", alpha=0.6) 

        plt.savefig(f"{output_path}/Combined/combined_analysis_{column}.png", bbox_inches="tight", dpi=300)
        plt.close()

def merge_csv_pair(file1, file2, output_file):
    """
    Takes in two files and an output path and merges the files.
    This is done by merging two matching cells of the files by the id,
    Merging unions the cells with reducing No Class Found if possible.

    Parameters:
        file1 (str): Path of a solution file to merge
        file2 (str): Path of a solution file to merge
        output_file (str) : Path whre the merged file should be saved in.
    
    """


    def unify_cell_values(val1, val2):
        """
        Takes in two cells and merges it accordingly.

        Parameters:
            val1 (str): Cell String 1
            val2 (str): Cell String 2
        
        """

        # Normalize missing values
        val1 = val1 if pd.notna(val1) else ""
        val2 = val2 if pd.notna(val2) else ""
        
        parts1 = set(part.strip() for part in val1.split("|") if part.strip())
        parts2 = set(part.strip() for part in val2.split("|") if part.strip())

        if parts1 == {"No Class Found"} and parts2 == {"No Class Found"}:
            return "No Class Found"

        # Merge and exclude "No Class Found" unless it"s the only entry in both
        merged = (parts1 | parts2) - {"No Class Found"}
        return "|".join(sorted(merged)) if merged else ""


    df1 = pd.read_csv(file1, dtype=str)
    df2 = pd.read_csv(file2, dtype=str)

    # Normalize column names
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    # Keep only rows with valid "id"
    df1 = df1[df1["id"].notna() & (df1["id"].str.len() > 4)]
    df2 = df2[df2["id"].notna() & (df2["id"].str.len() > 4)]

    # Set index to "id"
    df1.set_index("id", inplace=True)
    df2.set_index("id", inplace=True)

    # Align all ids and columns
    all_ids = df1.index.union(df2.index)
    all_columns = df1.columns.union(df2.columns)

    df1 = df1.reindex(index=all_ids, columns=all_columns)
    df2 = df2.reindex(index=all_ids, columns=all_columns)

    # Merge intelligently
    merged = pd.DataFrame(index=all_ids, columns=all_columns)

    for col in all_columns:
        merged[col] = [
            unify_cell_values(v1, v2)
            for v1, v2 in zip(df1[col], df2[col])
        ]

    # Reset index and write to output
    merged = merged.reset_index()
    merged.to_csv(output_file, index=False)

def create_id_subset():
    """
    Creates a csv file that has a subset of 1000 video id from the file solution_clip.
    The subset is created in a way, that every year (2019-2022) has 250 videos.
    The subset is saved under "random_ids_by_year.csv".
    """
  
    def extract_year(id_str):
        """
        Extracts the year from an ID string that ends with a date in the format YYYY-MM-DD.

        Parameters:
            id_str (str): id string

        Returns:
            string of id. E.g. "2019"
        """
        match = re.search(r"_(\d{4}-\d{2}-\d{2})$", str(id_str))
        if match:
            return int(match.group(1).split("-")[0])
        return 
    
    df = pd.read_csv(solution_clip)
    # Add year column and filter valid years
    df["year"] = df["id"].apply(extract_year)
    df = df[df["year"].between(2019, 2022)]

    # Get 250 random IDs from each year (or all if fewer than 250)
    selected_ids = []
    for year in range(2019, 2023):
        year_ids = df[df["year"] == year]["id"]
        sample_size = min(250, len(year_ids))
        
        if sample_size > 0:
            sampled = year_ids.sample(n=sample_size)
            selected_ids.extend(sampled.tolist())
            print(f"Selected {sample_size} IDs for year {year}")
        else:
            print(f"No IDs found for year {year}")

    # Create new DataFrame and save
    new_df = pd.DataFrame({"id": selected_ids})
    new_df.to_csv("random_ids_by_year.csv", index=False)
    print(f"Created file with {len(new_df)} total IDs")

def load_dataset_with_duplicates(main_csv_path: str) -> pd.DataFrame:
    """
    Takes in a path of a solution file and aggregates it with duplicates.
    Only duplicates with a original in the dataset from the solution file get added.
    Duplicates recieve the same classification result as their original.

    Parameters:
        main_csv_path (str): Path to the main dataset CSV file.

    Returns:
        pd.DataFrame: A combined DataFrame including both original and duplicate videos.
    """

    df_main = pd.read_csv(main_csv_path)
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates_to_originals.csv")

    # Filter to only valid mappings
    df_dups_valid = df_dups[df_dups["original_id"].isin(df_main["id"])]

    # Giving duplcates data from the original
    df_duplicates = df_dups_valid.merge(
        df_main, left_on="original_id", right_on="id", suffixes=("", "_original")
    )

    # Replaces wrong column names
    df_duplicates = df_duplicates.drop(columns=["id"])
    df_duplicates = df_duplicates.rename(columns={"duplicate_id": "id"})

    # Combine datasets
    df_combined = pd.concat([df_main, df_duplicates], ignore_index=True)

    return df_combined

def trend_graph(path):
    """
    Takes a solution file path and visualizies the content of each column in a line graph.
    This trend graph is saved under the output_path/Trend/modelname_column_trend.png

    Parameters:
        path (str): Path of solution file.
    """

    # Select Fontsize and Colors
    FONTSIZE = 24
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"
    ]

    # Get the name for the output
    name = path.split("/")[-1].split(".")[0].split("_")[0]

    # Aggregating the df with duplicates
    df = load_dataset_with_duplicates(path)

    # Extract date from ID and compute quarter
    df["date"] = pd.to_datetime(df["id"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["quarter"] = df["quarter"].str.replace(r"(\d{4})Q", r"\1-Q", regex=True)

    
    for category in ["animals", "climateactions", "consequences", "setting", "type"]:
        df_col = df[["quarter", category]].copy()

        # Split on "|" only
        df_col[category] = df_col[category].astype(str).str.split("|")
        df_col = df_col.explode(category)
        df_col[category] = df_col[category].str.strip()

        # Remove "No" only for certain columns
        if category in ["animals", "climateactions", "consequences"]:
            df_col = df_col[(df_col[category] != "") & (df_col[category].str.lower() != "no")]
        else:
            df_col = df_col[df_col[category] != ""]  # Keep everything else, including "No Setting"

        # Count occurrences per quarter and category
        count_df = df_col.groupby(["quarter", category]).size().reset_index(name="count")

        # Total valid entries per quarter
        total_per_quarter = df_col.groupby("quarter").size().reset_index(name="total")

        # Merge and calculate relative frequency
        merged = count_df.merge(total_per_quarter, on="quarter")
        merged["relative"] = merged["count"] / merged["total"]

        # Pivot for plotting
        pivot = merged.pivot(index="quarter", columns=category, values="relative").fillna(0)
        pivot = pivot.sort_index()

        # Plot
        ax = pivot.plot(marker="o", figsize=(22, 14), fontsize=FONTSIZE, color=colors)
        ax.set_ylabel("Relative Frequency", fontsize=FONTSIZE)
        ax.set_xlabel("Quarter", fontsize=FONTSIZE)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=FONTSIZE)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=FONTSIZE)
        ax.grid(True)

        plt.savefig(f"{output_path}/Trend/{name}_{category}_trend.png", bbox_inches="tight", dpi=300)
        plt.close()

def compare_assignments(file1_path, file2_path, target_value):
    """
    Calculates the jaccard similarity between two files and a taget-value.
    The column(s) needs to be asigned in code.
    Result will be printed.

    Parameters:
        file1_path (str): Path of first file
        file2_path (str): Path of second file
        target_value (str): Value of the jaccard similarity
    """

    # Columns to compare
    columns = ["type"]

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge on "id" (inner join to keep only matching IDs)
    merged = pd.merge(df1, df2, on="id", suffixes=("_file1", "_file2"))

    results = {}
    for col in columns:
        col_file1 = f"{col}_file1"
        col_file2 = f"{col}_file2"

        # Booleans for "No" labels in each file
        no_1 = merged[col_file1] == target_value
        no_2 = merged[col_file2] == target_value

        # Calculate intersection and union
        both_no = (no_1 & no_2).sum()
        either_no = (no_1 | no_2).sum()

        # Calculate overlap percentage
        overlap = (both_no / either_no) * 100 if either_no > 0 else 0.0

        results[col] = {
            "overlap_percentage": overlap,
            "total_both_no": both_no,
            "total_either_no": either_no
        }

    # Print results
    for col, data in results.items():
        print(
            f"Column '{col}':\n"
            f"  - Overlap: {data['overlap_percentage']:.2f}% of {target_value} labels match.\n"
            f"  - Both {target_value}': {data['total_both_no']}, Either {target_value}: {data['total_either_no']}\n")
    
    for col, data in results.items():
        print(f"{target_value}:")
        print(f"{data["overlap_percentage"]:.2f}% ({data["total_both_no"]}/{data["total_either_no"]})")

def count_value_per_column(file_path: str, target_value: str) -> dict:
    """
    Counts occurrences of `target_value` in each column (except "id") of a CSV file.
    Results can also be printed out.

    Args:
        file_path: Path to the CSV file.
        target_value: Value to count.

    Returns:
        A dictionary with columns as keys and counts as values.
    """
    df = pd.read_csv(file_path)
    columns_to_check = [col for col in df.columns if col != "id"]
    result = {col: (df[col] == target_value).sum() for col in columns_to_check}

    # Optinal Print output
    # print(f"Counts of "{target_value}" per column:")
    # for col, count in result.items():
    #    print(f"- {col}: {count}")

    return result

def drop_duplicate_ids(input_file):
    """
    Drops duplicate ids in a file. This can be used if the solution file is corrupted.

    Parameters:
        input_file (str): path of the input file.
    """

    df = pd.read_csv(input_file, dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop rows with missing or too-short IDs
    if "id" not in df.columns:
        raise ValueError("No id column found.")

    df = df[df["id"].notna() & (df["id"].str.strip().str.len() > 4)]

    # Normalize IDs
    df["id"] = df["id"].str.strip().str.lower()

    # Drop duplicates by keeping the first occurrence
    df = df.groupby("id", as_index=False).first()

    # Save cleaned CSV
    df.to_csv(input_file, index=False)

def print_unique_values_with_percentages(path):
    """
    Prints the absolute and relative apperance of every unique value in a solution file.
    The results are printed for each column.

    Parameters:
        path (str): Path from file
    
    """

    df = pd.read_csv(path)

    for group in df.columns:
        if group == "id":
            continue
        value_counts = df[group].value_counts()
        percentages = (value_counts / len(df)) * 100

        # Print each unique value with its percentage
        print(f'Unique values in "{group}":')
        for value, percent in percentages.items():
            if percent > 1:
                print(f"- {value}: {percent:.1f}%")

def group_unique_videos_quarter(path, column, target_value, exclude_memes=False):
    """
    Prints out the original video id and the amount it or its duplicates are present in a quarter. 
    Only videos with a specific target value get selected. 
    Memes can be included or excluded. 

    Parameters:
        path (str): Path of the file 
        column (str):
        target_value (str):
        exclude_memes (bool): False for including videos labeled as Memes, True for excluding
    
    """

    # Load main dataset and aggregate it with duplciates, and get the duplicate and original file.
    df = load_dataset_with_duplicates(path)
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates_to_originals.csv")

    # Extract date and quarter from video ID
    df["date"] = pd.to_datetime(df["id"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["quarter"] = df["quarter"].str.replace(r"(\d{4})Q", r"\1-Q", regex=True)

    # Map each video to its original_id (via duplicates mapping)
    dup_to_orig = dict(zip(df_dups["duplicate_id"], df_dups["original_id"]))
    df["original_id"] = df["id"].map(dup_to_orig)  # Map known duplicates
    df["original_id"] = df["original_id"].fillna(df["id"])  # Non-duplicates map to themselves


    original_mask = ~df["id"].isin(df_dups["duplicate_id"])  # True if the video is not a duplicate
    # Base condition for finding target videos
    condition = (original_mask & df[column].astype(str).str.contains(target_value, na=False))
    
    # Add meme exclusion if requested
    if exclude_memes:
        condition = condition & (~df["type"].astype(str).str.contains("Meme", na=False))
    # Find all original video IDs with the target value (and optionally excluding memes)
    
    originals_with_target = df[condition]
    valid_original_ids = set(originals_with_target["id"])

    # Filter to only rows whose original_id is in the valid originals
    df_filtered = df[df["original_id"].isin(valid_original_ids)]

    # Group by quarter and original_id, and count usage
    grouped = df_filtered.groupby(["quarter", "original_id"]).size().reset_index(name="count")

    # Total number of videos per quarter (no filtering)
    total_per_quarter = df.groupby("quarter").size().to_dict()

    # Print results
    for quarter in sorted(grouped["quarter"].unique()):
        print(f"\nQuarter: {quarter} - Total videos in quarter: {total_per_quarter.get(quarter, 0)}")
        quarter_data = grouped[grouped["quarter"] == quarter]
        quarter_data = quarter_data.sort_values(by="count", ascending=False)
        for _, row in quarter_data.iterrows():
            if row["count"] > 2:
                print(f"{row["original_id"]}: {row["count"]}")
