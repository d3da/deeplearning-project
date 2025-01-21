import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


def create_dataset_csv(root_dir, output_train_csv, output_val_csv):
    """
    Create CSV files for training and validation from GTSRB dataset structure.

    Parameters:
    root_dir: str
        Path to GTSRB/Final_Training/Images directory
    output_train_csv: str
        Path to save training CSV
    output_val_csv: str
        Path to save validation CSV
    """
    data = []

    # Iterate through all class directories
    for class_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(class_dir):
            continue

        class_num = int(os.path.basename(class_dir))

        # Read class-specific CSV file
        csv_file = os.path.join(class_dir, f"GT-{os.path.basename(class_dir)}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, sep=";")

            # Create relative path and add class label
            df["path"] = df["Filename"].apply(
                lambda x: os.path.join(os.path.basename(class_dir), x)
            )
            df["label"] = class_num

            # Select only needed columns
            df = df[["path", "label"]]
            data.append(df)

    # Combine all data
    full_df = pd.concat(data, ignore_index=True)

    # Split into train and validation
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["label"]
    )

    # Save to CSV files
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)

    print(f"Created training dataset with {len(train_df)} images")
    print(f"Created validation dataset with {len(val_df)} images")
    print(f"Number of classes: {len(full_df['label'].unique())}")


def main():
    # Paths
    root_dir = "data/GTSRB/Final_Training/Images"
    output_train_csv = "data/train.csv"
    output_val_csv = "data/val.csv"

    # Create dataset CSV files
    create_dataset_csv(root_dir, output_train_csv, output_val_csv)


if __name__ == "__main__":
    main()
