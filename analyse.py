import datetime
import glob
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

diagram_output = "/work/mburmest/bachelorarbeit/Visualizations"
#TODO: Maybe some data clean_up

output_path = "/work/mburmest/bachelorarbeit/Diagrams/"

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


solutions = [solution_clip, 
             solution_videollava_1, solution_videollava_2, solutions_videollava_3, 
             solution_videochatgpt_1, solution_videochatgpt_2, solution_videochatgpt_3, 
             solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3]

exception_videollava = "/work/mburmest/bachelorarbeit/Exceptions/videollava_exception.csv"
exception_videochatgpt = "/work/mburmest/bachelorarbeit/Exceptions/videochatgpt_exception.csv"
exception_pandagpt = "/work/mburmest/bachelorarbeit/Exceptions/pandagpt_exception.csv"

exceptions_files = [exception_videollava, exception_videochatgpt, exception_pandagpt]


def exceptions():
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

# Get all diagrams for each main category and subcategories. 
"""
animals = ["Pets", "Farm animals", "Polar bears", "Land mammals", "Sea mammals", "Fish", "Amphibians", "Reptiles", "Invertebrates", "Birds", "Insects", "Other animals", "No", "FAILED YES/NO", "NO CLASS FOUND"]
consequences = ["Floods", "Drought", "Wildfires", "Rising temperature", "Other extreme weather events", "Melting ice", "Sea level rise", "Human rights", "Economic consequences", "Biodiversity loss", "Covid", "Health", "Other consequence", "No", "FAILED YES/NO", "NO CLASS FOUND"]
climate_actions = ["Politics", "Protests", "Solar energy", "Wind energy", "Hydropower", "Bioenergy", "Coal", "Oil", "Natural gas", "Other climate action", "No", "FAILED YES/NO", "NO CLASS FOUND"]
settings = ["No setting", "Residential area", "Industrial area", "Commercial area", "Agricultural", "Rural", "Indoor space", "Arctic", "Antarctica", "Ocean", "Coastal", "Desert", "Forest", "Jungle", "Other nature", "Outer space", "Other setting", "No", "FAILED YES/NO", "NO CLASS FOUND"]
types = ["Event invitations", "Meme", "Infographic", "Data visualization", "Illustration", "Screenshot", "Single photo", "Photo collage", "Other type", "No", "FAILED YES/NO", "NO CLASS FOUND"]
"""
def info_subset():
    videos = glob.glob("/ceph/lprasse/ClimateVisions/Videos/*/*/*.mp4")
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv")
    duplicates = df_dups['id'].astype(str)

    print(f"Videos total: {len(videos)}")

    # Create a DataFrame from the video paths
    df_videos = pd.DataFrame({'path': videos})

    # Extract ID and date from filenames using regex
    # Example filename: id_1228086819004321798_2020-02-14.mp4
    df_videos['filename'] = df_videos['path'].str.extract(r'([^/]+\.mp4)$')
    df_videos['id'] = df_videos['filename'].str.extract(r'(id_\d+_\d{4}-\d{2}-\d{2})')
    df_videos['date'] = df_videos['id'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    df_videos['year'] = pd.to_datetime(df_videos['date'], errors='coerce').dt.year

    # Remove duplicates
    df_filtered = df_videos[~df_videos['id'].isin(duplicates)]

    print(f"Videos after duplicates: {len(df_filtered)}")

    # Count videos per year (for 2019–2022 only)
    for year in [2019, 2020, 2021, 2022]:
        count = df_filtered[df_filtered['year'] == year].shape[0]
        print(f"Videos in {year}: {count}")

def same_subset(paths):
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
    def capitalize_words(text):
        if pd.isna(text):  # Skip NaN values
            return text
        # Process each part separated by |
        parts = str(text).split("|")
        capitalized_parts = []
        for part in parts:
            # Capitalize each word in the part
            capitalized_part = " ".join(word.capitalize() for word in part.split())
            capitalized_parts.append(capitalized_part)
        # Rejoin with | separator
        return "|".join(capitalized_parts)

    df = pd.read_csv(path)
    df = df.map(capitalize_words)
    df['id'] = df['id'].str.lower()
    #df = sort_ids(df)

    original_path = Path(path)
    new_path = original_path.parent / "NEW" / original_path.name

    df.to_csv(new_path, index=False)

def fix_animals_column(df):
    def replace_other(cell):
        if pd.isna(cell):
            return cell
        parts = [p.strip() for p in str(cell).split("|")]
        parts = ["Other Animals" if p == "Other" else p for p in parts]
        return "|".join(parts)

    df["animals"] = df["animals"].apply(replace_other)
    return df

def merge_categories(df):
    def fix_setting(cell):
        if pd.isna(cell):
            return cell
        parts = [p.strip() for p in str(cell).split("|")]
        if "Forest" not in parts and "Jungle" not in parts:
            return "|".join(parts)
        # Remove both if either is present, then add combined
        parts = [p for p in parts if p not in {"Forest", "Jungle"}]
        parts.append("Forest, Jungle")
        return "|".join(parts)

    df["setting"] = df["setting"].apply(fix_setting)
    return df

def sort_ids(df):
    df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
    df["numeric_id"] = df["id"].str.split("_").str[1].astype(int)
    return df.sort_values(by=["date", "numeric_id"]).drop(columns=["date", "numeric_id"])

def visualize(path):
    PLOT_COLS = ["animals", "climateactions", "consequences", "setting", "type"]
    all_data = {col: defaultdict(dict) for col in PLOT_COLS}

    df = pd.read_csv(path)
    exceptions_ids = exceptions()
    df = df[~df["id"].isin(exceptions_ids)]
    name = path.split("/")[-1].split(".")[0]

    fontsize_global = 16

    for category in PLOT_COLS:
        
        split_categories = df[category].astype(str).str.split("|").explode().str.strip()
        split_categories = split_categories[(split_categories != "") & (split_categories != "nan")]

        category_counts = split_categories.value_counts().sort_values(ascending=False)
        category_percents = ((category_counts / len(df)) * 100).round(1)

        # Store for later multi-dataset plot
        for label in category_counts.index:
            all_data[category][label][name] = category_counts[label]

        # Print data table
        print(f"\n{name} - {category} Analysis:")
        result_df = pd.DataFrame({
            "Category": category_counts.index,
            "Count": category_counts.values,
            "Percentage": category_percents.values
        })
        print(result_df.to_string(index=False))

        plt.rcParams.update({'font.size': fontsize_global})

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
        #ax1.set_title(f"{name} - {category} (Count & Percentage Labels)")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Rotate x-tick labels
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels(x, rotation=45, ha="right", fontsize=fontsize_global)

        # Annotate bars with count + percentage
        for i, (bar, pct) in enumerate(zip(bars, percents)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:,}\n{pct}%", ha="center", va="bottom", fontsize=fontsize_global)

        #fig.tight_layout()
        plt.savefig(f"{output_path}{name}_{category}.png", bbox_inches="tight", dpi=300)
        plt.close()

def combined_bar_charts(csv_paths):
    PLOT_COLS = ["type"]
    fontsize_global = 18  # Consistent font size across plots
    custom_colors = ["teal", "orange", "purple", "green", "crimson"]  

    all_data = {col: defaultdict(dict) for col in PLOT_COLS}
    dfs = [pd.read_csv(path) for path in csv_paths]
    names = [path.split("/")[-1].split(".")[0].split("_")[0] for path in csv_paths]

    # Step 1: Create a set of blacklisted IDs (rows with "No Class Found" in any PLOT_COLS column)
    blacklisted_ids = set()
    for df in dfs:
        for col in PLOT_COLS:
            mask = df[col].astype(str).str.contains("No Class Found", na=False)            

            blacklisted_ids.update(df.loc[mask, "id"].tolist())

    # Step 2: Filter out blacklisted rows from all DataFrames
    dfs = [df[~df["id"].isin(blacklisted_ids)].copy() for df in dfs]

    # Set global font size
    plt.rcParams.update({'font.size': fontsize_global})

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
        width = 0.90 / len(dfs)  # width per bar

        fig, ax = plt.subplots(figsize=(24, 7))
        for i, file_counts in enumerate(data):
            color = custom_colors[i % len(custom_colors)]  # Safeguard against overflow
            bar = ax.bar(x + i * width, file_counts, width, color=color, label=names[i])

            # 🔧 Count + Percentage label — editable section
            total = sum(file_counts)
            for idx, height in enumerate(file_counts):
                percent = (height / total * 100) if total > 0 else 0
                ax.text(
                    x[idx] + i * width,
                    height,
                    f"{percent:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize_global - 4,
                    rotation=0,
                )
            # End editable section

        ax.set_ylabel("Count", fontsize=fontsize_global)
        ax.set_xticks(x + width * (len(dfs) - 1) / 2)
        ax.set_xticklabels(all_categories, rotation=45, ha="right", fontsize=fontsize_global)
        ax.legend(fontsize=fontsize_global)
        ax.grid(axis="y", linestyle="--", alpha=0.6) 

        plt.savefig(f"{output_path}/Combined/combined_analysis_{column}.png", bbox_inches="tight", dpi=300)
        plt.close()

def merge_csv_pair(file1, file2, output_file):
    def unify_cell_values(val1, val2):
    # Normalize missing values
        val1 = val1 if pd.notna(val1) else ''
        val2 = val2 if pd.notna(val2) else ''
        
        parts1 = set(part.strip() for part in val1.split('|') if part.strip())
        parts2 = set(part.strip() for part in val2.split('|') if part.strip())

        if parts1 == {'No Class Found'} and parts2 == {'No Class Found'}:
            return 'No Class Found'

        # Merge and exclude 'No Class Found' unless it's the only entry in both
        merged = (parts1 | parts2) - {'No Class Found'}
        return '|'.join(sorted(merged)) if merged else ''


    df1 = pd.read_csv(file1, dtype=str)
    df2 = pd.read_csv(file2, dtype=str)

    # Normalize column names
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    # Keep only rows with valid "id"
    df1 = df1[df1["id"].notna() & (df1["id"].str.len() > 4)]
    df2 = df2[df2["id"].notna() & (df2["id"].str.len() > 4)]

    # Set index to 'id'
    df1.set_index('id', inplace=True)
    df2.set_index('id', inplace=True)

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
  
    def extract_year(id_str):
        match = re.search(r'_(\d{4}-\d{2}-\d{2})$', str(id_str))
        if match:
            return int(match.group(1).split('-')[0])
        return 
    
    df = pd.read_csv(solution_clip)
    # Add year column and filter valid years
    df['year'] = df['id'].apply(extract_year)
    df = df[df['year'].between(2019, 2022)]  # Only keep 2019-2022

    # Get 250 random IDs from each year (or all if fewer than 250)
    selected_ids = []
    for year in range(2019, 2023):
        year_ids = df[df['year'] == year]['id']
        sample_size = min(250, len(year_ids))
        
        if sample_size > 0:
            sampled = year_ids.sample(n=sample_size)
            selected_ids.extend(sampled.tolist())
            print(f"Selected {sample_size} IDs for year {year}")
        else:
            print(f"No IDs found for year {year}")

    # Create new DataFrame and save
    new_df = pd.DataFrame({'id': selected_ids})
    new_df.to_csv('random_ids_by_year.csv', index=False)
    print(f"Created file with {len(new_df)} total IDs")

def load_dataset_with_duplicates(main_csv_path: str) -> pd.DataFrame:
    # Step 1: Read the main CSV
    df_main = pd.read_csv(main_csv_path)

    # Step 2: Read the duplicates CSV
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates_to_originals.csv")

    # Step 3: Filter to only valid mappings
    df_dups_valid = df_dups[df_dups['original_id'].isin(df_main['id'])]

    # Step 4: Merge to get all duplicate metadata from originals in one go
    df_duplicates = df_dups_valid.merge(
        df_main, left_on='original_id', right_on='id', suffixes=('', '_original')
    )

    # Step 5: Replace 'id' column with the duplicate_id
    df_duplicates = df_duplicates.drop(columns=['id'])
    df_duplicates = df_duplicates.rename(columns={'duplicate_id': 'id'})

    # Step 6: Combine
    df_combined = pd.concat([df_main, df_duplicates], ignore_index=True)

    return df_combined

def trend_graph(path):
    FONTSIZE = 16
    df = load_dataset_with_duplicates(path)

    # Extract date from ID and compute quarter
    df['date'] = pd.to_datetime(df['id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)
    df['quarter'] = df['quarter'].str.replace(r'(\d{4})Q', r'\1-Q', regex=True)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"
    ]

    for category in ["animals", "climateactions", "consequences", "setting", "type"]:
        df_col = df[['quarter', category]].copy()

        # Split on '|' only
        df_col[category] = df_col[category].astype(str).str.split('|')
        df_col = df_col.explode(category)
        df_col[category] = df_col[category].str.strip()

        # Remove 'No' only for certain columns
        if category in ["animals", "climateactions", "consequences"]:
            df_col = df_col[(df_col[category] != '') & (df_col[category].str.lower() != 'no')]
        else:
            df_col = df_col[df_col[category] != '']  # Keep everything else, including "No Setting"

        # Count occurrences per quarter and category
        count_df = df_col.groupby(['quarter', category]).size().reset_index(name='count')

        # Total valid entries per quarter
        total_per_quarter = df_col.groupby('quarter').size().reset_index(name='total')

        # Merge and calculate relative frequency
        merged = count_df.merge(total_per_quarter, on='quarter')
        merged['relative'] = merged['count'] / merged['total']

        # Pivot for plotting
        pivot = merged.pivot(index='quarter', columns=category, values='relative').fillna(0)
        pivot = pivot.sort_index()

        # Plot
        ax = pivot.plot(marker='o', figsize=(22, 14), fontsize=FONTSIZE, color=colors)
        ax.set_ylabel("Relative Frequency", fontsize=FONTSIZE)
        ax.set_xlabel("Quarter", fontsize=FONTSIZE)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=FONTSIZE)
        ax.tick_params(axis='y', labelsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE, title=category.capitalize(),loc='upper right')
        ax.grid(True)

        plt.savefig(f"{output_path}/Trend/clip_{category}_trend.png", bbox_inches="tight", dpi=300)
        plt.close()

def compare_assignments(file1_path, file2_path, target_value):
    #Jaccard Similarity
    # Load both files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge on 'id' (inner join to keep only matching IDs)
    merged = pd.merge(df1, df2, on='id', suffixes=('_file1', '_file2'))

    # Columns to compare
    columns = ["type"]

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
            'overlap_percentage': overlap,
            'total_both_no': both_no,
            'total_either_no': either_no
        }

    # Print results
    for col, data in results.items():
        print(
            f"Column '{col}':\n"
            f"  - Overlap: {data['overlap_percentage']:.2f}% of {target_value} labels match.\n"
            f"  - Both {target_value}': {data['total_both_no']}, Either {target_value}: {data['total_either_no']}\n"
        )
    
    for col, data in results.items():
        print(f"{target_value}:")
        print(f"{data['overlap_percentage']:.2f}\% ({data['total_both_no']}/{data['total_either_no']})")

def count_value_per_column(file_path: str, target_value: str) -> dict:
    """
    Counts occurrences of `target_value` in each column (except 'id') of a CSV file.

    Args:
        file_path: Path to the CSV file.
        target_value: Value to count.

    Returns:
        A dictionary with columns as keys and counts as values.
    """
    df = pd.read_csv(file_path)
    columns_to_check = [col for col in df.columns if col != 'id']
    result = {col: (df[col] == target_value).sum() for col in columns_to_check}

    #print(f"Counts of '{target_value}' per column:")
    #for col, count in result.items():
        #print(f"- {col}: {count}")

    return result

def drop_duplicate_ids(input_file):
    df = pd.read_csv(input_file, dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop rows with missing or too-short IDs
    if "id" not in df.columns:
        raise ValueError("No 'id' column found.")

    df = df[df["id"].notna() & (df["id"].str.strip().str.len() > 4)]

    # Normalize IDs
    df["id"] = df["id"].str.strip().str.lower()

    # Drop duplicates by keeping the first occurrence
    df = df.groupby("id", as_index=False).first()

    # Save cleaned CSV
    df.to_csv(input_file, index=False)

def print_unique_values_with_percentages(path):
    df = pd.read_csv(path)

    for group in df.columns:
        if group == "id":
            continue

        
        value_counts = df[group].value_counts()
        percentages = (value_counts / len(df)) * 100

        # Print each unique value with its percentage
        print(f"Unique values in '{group}':")
        for value, percent in percentages.items():
            if percent > 1:
                print(f"- {value}: {percent:.1f}%")

def print_amount_values(path):
    df = load_dataset_with_duplicates(path)

    # Extract date from ID and compute quarter
    df['date'] = pd.to_datetime(df['id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)
    df['quarter'] = df['quarter'].str.replace(r'(\d{4})Q', r'\1-Q', regex=True)

    df["quarter"].value_counts()

def group_unique_videos_quarter(path, column, target_value, exclude_memes=False):
    # Load main dataset and duplicates mapping
    df = load_dataset_with_duplicates(path)
    df_dups = pd.read_csv("/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates_to_originals.csv")

    # Extract date and quarter from video ID
    df['date'] = pd.to_datetime(df['id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)
    df['quarter'] = df['quarter'].str.replace(r'(\d{4})Q', r'\1-Q', regex=True)

    # Map each video to its original_id (via duplicates mapping)
    dup_to_orig = dict(zip(df_dups['duplicate_id'], df_dups['original_id']))
    df['original_id'] = df['id'].map(dup_to_orig)  # Map known duplicates
    df['original_id'] = df['original_id'].fillna(df['id'])  # Non-duplicates map to themselves


    original_mask = ~df['id'].isin(df_dups['duplicate_id'])  # True if the video is not a duplicate
    # Base condition for finding target videos
    condition = (original_mask & df[column].astype(str).str.contains(target_value, na=False))
    
    # Add meme exclusion if requested
    if exclude_memes:
        condition = condition & (~df['type'].astype(str).str.contains('Meme', na=False))
    # Find all original video IDs with the target value (and optionally excluding memes)
    
    originals_with_target = df[condition]
    valid_original_ids = set(originals_with_target['id'])

    # Filter to only rows whose original_id is in the valid originals
    df_filtered = df[df['original_id'].isin(valid_original_ids)]

    # Group by quarter and original_id, and count usage
    grouped = df_filtered.groupby(['quarter', 'original_id']).size().reset_index(name='count')

    # Total number of videos per quarter (no filtering)
    total_per_quarter = df.groupby('quarter').size().to_dict()

    # Print results
    for quarter in sorted(grouped['quarter'].unique()):
        print(f"\nQuarter: {quarter} - Total videos in quarter: {total_per_quarter.get(quarter, 0)}")
        quarter_data = grouped[grouped['quarter'] == quarter]
        quarter_data = quarter_data.sort_values(by='count', ascending=False)
        for _, row in quarter_data.iterrows():
            if row['count'] > 2:
                print(f"{row['original_id']}: {row['count']}")




#for x in ["All", "Data Visualization", "Event Invitations", "Illustration", "Infographic", "Meme", "Other Type", "Photo Collage", "Screenshot", "Single Photo"]:
    #compare_assignments(solution_videochatgpt_3, solution_pandagpt_3, x)
    #compare_assignments(solution_videochatgpt_3, solution_clip, x)
    #compare_assignments(solution_clip, solution_pandagpt_3, x)

#group_unique_videos_quarter(solution_clip, "animals", "Farm Animals", exclude_memes=False)
#trend_graph(solution_clip)
"""
for solution in [   solution_videollava_1, solution_videollava_2, solutions_videollava_3, 
                    solution_videochatgpt_1, solution_videochatgpt_2, solution_videochatgpt_3, 
                    solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3]:
    
    print(f"\n {solution}")
    no_class = count_value_per_column(solution, "No Class Found")
    no = count_value_per_column(solution, "No")

    for col in no_class:
        class_count = no_class[col]
        no_count = no[col]
        valid_total = 44927 - no_count

        if valid_total > 0:
            relative = class_count / valid_total
        else:
            relative = 0.0  # Avoid division by zero

        print(f"- {col}: {relative:.2%}")  # formatted as percentage
"""

#info_subset()
combined_bar_charts([solution_videochatgpt_3,solution_pandagpt_3 ,solution_clip])
#print_unique_values_with_percentages(solution_clip)
#visualize(solutions_videollava_3)
#visualize(solutions_videollava_1000)
#visualize(solution_pandagpt_3)
#visualize(solution_videochatgpt_3)
#visualize(solution_clip)
#visualize(solution_pandagpt_1000)
#combined_bar_charts([solutions_videollava_3, solutions_videollava_1000])
#create_id_subset()
#merge_csv_pair(solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3)
#drop_duplicate_ids(solution_videochatgpt_1)
#drop_duplicate_ids(solution_videochatgpt_2)


#merge_csv_pair(solution_videochatgpt_1,solution_videochatgpt_2,"/work/mburmest/bachelorarbeit/Solution/videochatgpt_solution_4.csv")
#merge_csv_pair(solution_videollava_1, solution_videollava_2, solutions_videollava_3)

# Put the bars for each path together in a different color


#same_subset(solutions)

#for solution in solutions:
    
#trend_graph(solution_clip)

#compare_no_assignments(solutions_videollava_3, solutions_videollava_1000)

# Example usage

#counts = count_value_per_column(solution_videochatgpt_3, "No Class Found")
#counts = count_value_per_column("/work/mburmest/bachelorarbeit/Solution/videochatgpt_solution_4.csv", "No Class Found")

"""
    original_path = Path(solution)
    new_path = original_path.parent / "NEW" / original_path.name

    df = pd.read_csv(solution)
    df = fix_animals_column(df)
    df.to_csv(new_path, index = False)
 """
