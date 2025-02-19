import pandas as pd
from tqdm import tqdm

def convert_to_level(dataset, conversion_dict):
    """
    Updates each entity's label in the dataset using the conversion_dict.
    If a label is not found in conversion_dict, it remains unchanged.
    """
    updated_dataset = []
    print(dataset)
    for entry in tqdm(dataset, desc="Processing entries"):
        new_entry = entry.copy()
        new_entry["entities"] = [
            {**entity, "Label": conversion_dict.get(entity["Label"], entity["Label"])}
            for entity in entry["entities"]
        ]
        updated_dataset.append(new_entry)
    return updated_dataset


def collapse_leaves_one_level(df):
    """
    Collapses one 'leaf' level of the graph.

    Returns:
        df_collapsed (DataFrame): DataFrame with leaf edges removed.
        mapping (dict): Mapping from collapsed child to parent.
    """
    leaves = set(df['child']) - set(df['parent'])
    df_collapsed = df[~df['child'].isin(leaves)]
    child_parent = dict(zip(df[df['child'].isin(leaves)]['child'], df[df['child'].isin(leaves)]['parent']))
    return df_collapsed, child_parent

def multi_level_collapse(df, train, validation, test, iterations=1):
    """
    Iteratively collapses leaves in the graph and updates train, validation,
    and test datasets accordingly.

    Args:
        df (DataFrame): The initial graph DataFrame.
        train_data, validation_data, test_data (list): Datasets to update.
        iterations (int): Number of collapse iterations.

    Returns:
        tuple: Lists of labels for each iteration for train, validation, and test datasets.
    """
    df_current = df.copy()
    
    for i in range(iterations):
        df_current, mapping = collapse_leaves_one_level(df_current)
        print(f"Iteration {i+1} - Collapsed {len(mapping)} leaves")
        
        # Update datasets using the current mapping
        train = convert_to_level(train, mapping)
        validation = convert_to_level(validation, mapping)
        test = convert_to_level(test, mapping)
    return train, validation,test