import datetime
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


solutions = [solution_clip, 
             solution_videollava_1, solution_videollava_2, solutions_videollava_3, 
             solution_videochatgpt_1, solution_videochatgpt_2, solution_videochatgpt_3, 
             solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3]

exception_videollava = "/work/mburmest/bachelorarbeit/Exceptions/videollava_exception.csv"
exception_videochatgpt = "/work/mburmest/bachelorarbeit/Exceptions/videochatgpt_exception.csv"
exception_pandagpt = "/work/mburmest/bachelorarbeit/Exceptions/pandagpt_exception.csv"

exceptions = [exception_videollava, exception_videochatgpt, exception_pandagpt]


def exceptions():
    exception_ids = set()
    for exception in exceptions:
        with open(exception) as f:
            ids = re.findall(r"id_\d+_\d{4}-\d{2}-\d{2}", f.read())
            print(f"{exception}: {len(ids)} IDs")

    for exception in exceptions:
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
    df = df[~df["id"].isin(exceptions())]
    name = path.split("/")[-1].split(".")[0]

    fontsize_global = 16

    for category in PLOT_COLS:
        

        split_categories = df[category].astype(str).str.split("|").explode().str.strip()
        split_categories = split_categories[(split_categories != "") & (split_categories != "nan")]

        category_counts = split_categories.value_counts().sort_values(ascending=False)
        category_percents = (category_counts / category_counts.sum() * 100).round(1)

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

        if path == solutions_videollava_1000:
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
    PLOT_COLS = ["animals", "climateactions", "consequences", "setting", "type"]
    
    all_data = {col: defaultdict(dict) for col in PLOT_COLS}
    dfs = [pd.read_csv(path) for path in csv_paths]
    names = [path.split("/")[-1].split(".")[0] for path in csv_paths]

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
        width = 0.8 / len(dfs)  # width per bar

        fig, ax = plt.subplots(figsize=(18, 7))
        for i, file_counts in enumerate(data):
            bar = ax.bar(x + i * width, file_counts, width, label=names[i])

            # 🔧 Count + Percentage label — editable section
            total = sum(file_counts)
            for idx, height in enumerate(file_counts):
                percent = (height / total * 100) if total > 0 else 0
                ax.text(
                    x[idx] + i * width,
                    height,
                    f"{int(height)}\n({percent:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )
            # End editable section

        ax.set_title(f"Distribution of categories in column: {column}")
        ax.set_xlabel("Categories")
        ax.set_ylabel("Count")
        ax.set_xticks(x + width * (len(dfs) - 1) / 2)
        ax.set_xticklabels(all_categories, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6) 


        plt.savefig(f"{output_path}combined_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()

def merge_csv_pair(file1, file2, output_file):
    df1 = pd.read_csv(file1,dtype=str)
    df2 = pd.read_csv(file2,dtype=str)

    # Strip column names and convert to lowercase
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    # Drop fully empty or junk rows
    df1 = df1.dropna(how="any")
    df2 = df2.dropna(how="any")

    # Keep only rows with a proper "id"
    df1 = df1[df1["id"].notna() & (df1["id"].str.len() > 4)]
    df2 = df2[df2["id"].notna() & (df2["id"].str.len() > 4)]

    # Union of all IDs
    all_ids = df1.index.union(df2.index)

    # Align both DataFrames
    df1 = df1.reindex(all_ids)
    df2 = df2.reindex(all_ids)

    # Ensure consistent column structure
    if not df1.columns.equals(df2.columns):
        raise ValueError("Files have mismatched columns.")

    # Merge intelligently
    merged = df1.copy()
    for col in df1.columns:
        merged[col] = df1[col].where(
            (df1[col] != "No Class Found") & df1[col].notna(),
            df2[col]
        )

    # Final cleanup
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

    # Extract full date from 'id' and get quarter
    df['date'] = pd.to_datetime(df['id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)  # e.g., '2019Q1', '2019Q2'

    columns = ["animals", "climateactions", "consequences", "type"]
    replacements = {"No": None}

    for category in columns:
        df_col = df.copy()
        df_col[category] = df_col[category].replace(replacements)
        df_col[category] = df_col[category].dropna().str.split(r'\||, ')
        df_col = df_col.explode(category).dropna()

        # Count category occurrences by quarter
        count_df = df_col.groupby(['quarter', category]).size().reset_index(name='count')

        # Total number of videos per quarter
        total_per_quarter = df.groupby('quarter').size().rename("total").reset_index()

        # Compute relative frequency
        merged = count_df.merge(total_per_quarter, on='quarter')
        merged['relative'] = merged['count'] / merged['total']

        # Pivot for plotting
        pivot = merged.pivot(index='quarter', columns=category, values='relative').fillna(0)

        # Ensure sorted order
        pivot = pivot.sort_index()

        # Plot with font settings
        ax = pivot.plot(marker='o', figsize=(14, 8), fontsize=FONTSIZE)
        ax.set_ylabel("Relative Frequency", fontsize=FONTSIZE)
        ax.set_xlabel("Quarter", fontsize=FONTSIZE)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=FONTSIZE)
        ax.tick_params(axis='y', labelsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE)
        ax.grid(True)

        plt.savefig(f"{output_path}/Trend/{category}_trend_1.png", bbox_inches="tight", dpi=300)
        plt.close()

def compare_no_assignments(file1_path, file2_path):
    # Load both files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge on 'id' (inner join to keep only matching IDs)
    merged = pd.merge(df1, df2, on='id', suffixes=('_file1', '_file2'))

    # Columns to compare (all except 'id')
    columns = ['animals', 'climateactions', 'consequences', 'setting', 'type']

    results = {}
    for col in columns:
        col_file1 = f"{col}_file1"
        col_file2 = f"{col}_file2"

        # Count "No" in each file for the column
        no_file1 = (merged[col_file1] == 'No').sum()
        no_file2 = (merged[col_file2] == 'No').sum()

        # Rows where both files have "No"
        both_no = ((merged[col_file1] == 'No') & (merged[col_file2] == 'No')).sum()

        # Determine which file has more "No" values
        if no_file1 > no_file2:
            denominator = no_file1
            file_with_more = "File 1"
        elif no_file2 > no_file1:
            denominator = no_file2
            file_with_more = "File 2"
        else:
            denominator = no_file1  # or no_file2 (they're equal)
            file_with_more = "Both files have equal 'No' counts"

        # Calculate similarity percentage (avoid division by zero)
        similarity = (both_no / denominator) * 100 if denominator > 0 else 0.0

        results[col] = {
            'similarity_percentage': similarity,
            'file_with_more_no': file_with_more,
            'total_matching_no': both_no,
            'total_no_in_larger_file': denominator
        }

    # Print results
    for col, data in results.items():
        print(
            f"Column '{col}':\n"
            f"  - Similarity: {data['similarity_percentage']:.2f}% of 'No' assignments match.\n"
            f"  - File with more 'No': {data['file_with_more_no']} "
            f"({data['total_no_in_larger_file']} vs {data['total_matching_no']} matching)\n"
        )

def count_value_per_column(file_path: str, target_value: str) -> dict:
    """
    Counts occurrences of `target_value` in each column (except 'id') of a CSV file.

    Args:
        file_path: Path to the CSV file.
        target_value: Value to count (e.g., "No").

    Returns:
        A dictionary with columns as keys and counts as values.
    """
    df = pd.read_csv(file_path)
    columns_to_check = [col for col in df.columns if col != 'id']
    result = {col: (df[col] == target_value).sum() for col in columns_to_check}

    print(f"Counts of '{target_value}' per column:")
    for col, count in result.items():
        print(f"- {col}: {count}")

    return result


#visualize(solutions_videollava_3)
#visualize(solutions_videollava_1000)
#visualize(solution_pandagpt_3)
#visualize(solution_videochatgpt_3)
#visualize(solution_clip)
#combined_bar_charts([solutions_videollava_3, solutions_videollava_1000])
#create_id_subset()
#merge_csv_pair(solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3)
#merge_csv_pair(solution_videochatgpt_1,solution_videochatgpt_2,solution_videochatgpt_3)
#merge_csv_pair(solution_videollava_1, solution_videollava_2, solutions_videollava_3)

# Put the bars for each path together in a different color


#same_subset(solutions)
#for solution in solutions:
    
#trend_graph(solution_clip)

#compare_no_assignments(solutions_videollava_3, solutions_videollava_1000)

df = pd.read_csv(solution_videochatgpt_3)
print("ADw")
df = df.reset_index(drop=True)
#df = df.drop(columns=["index"])
# Example usage

#counts = count_value_per_column(solutions_videollava_3, "No")

"""
    original_path = Path(solution)
    new_path = original_path.parent / "NEW" / original_path.name

    df = pd.read_csv(solution)
    df = fix_animals_column(df)
    df.to_csv(new_path, index = False)
 """
