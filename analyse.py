import datetime
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


# Get all exceptions
exception_ids = set()
for exception in exceptions:
    with open(exception) as f:
        ids = re.findall(r"id_\d+_\d{4}-\d{2}-\d{2}", f.read())
        print(f"{exception}: {len(ids)} IDs")

for exception in exceptions:
    with open(exception) as f:
        exception_ids.update(re.findall(r"id_\d+_\d{4}-\d{2}-\d{2}", f.read()))

print(f"\nTotal exception_ids: {len(exception_ids)}")

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
    df = df[~df["id"].isin(exception_ids)]
    name = path.split("/")[-1].split(".")[0]

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

        # Combined bar chart (only bars for count, percentage as text labels)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        x = category_counts.index
        counts = category_counts.values
        percents = category_percents.values

        bars = ax1.bar(x, counts, color="teal", label="Count")
        ax1.set_ylabel("Count")
        ax1.set_title(f"{name} - {category} (Count & Percentage Labels)")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Rotate x-tick labels
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels(x, rotation=45, ha="right")

        # Annotate bars with count + percentage
        for i, (bar, pct) in enumerate(zip(bars, percents)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:,}\n({pct}%)", ha="center", va="bottom", fontsize=8)

        #fig.tight_layout()
        plt.savefig(f"{output_path}{name}_{category}_combined_analysis.png", bbox_inches="tight", dpi=300)
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

def merge_csv_pair(file1, file2, output_file):
    """Merge two CSVs row-wise by ID, preferring non-"NO CLASS FOUND" values and skipping broken rows."""
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

    # Drop duplicates by "id"
    df1 = df1.drop_duplicates(subset="id").set_index("id")
    df2 = df2.drop_duplicates(subset="id").set_index("id")

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
            (df1[col] != "NO CLASS FOUND") & df1[col].notna(),
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

#create_id_subset()
#merge_csv_pair(solution_pandagpt_1, solution_pandagpt_2, solution_pandagpt_3)
#merge_csv_pair(solution_videochatgpt_1,solution_videochatgpt_2,solution_videochatgpt_3)
#merge_csv_pair(solution_videollava_1, solution_videollava_2, solutions_videollava_3)

# Put the bars for each path together in a different color


#same_subset(solutions)
#for solution in solutions:
 #   visualize(solution)

"""
    original_path = Path(solution)
    new_path = original_path.parent / "NEW" / original_path.name

    df = pd.read_csv(solution)
    df = fix_animals_column(df)
    df.to_csv(new_path, index = False)
 """
