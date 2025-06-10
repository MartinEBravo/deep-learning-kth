import os


def create_folder(folder_name):
    """
    Creates a new folder if it doesn't already exist.

    Args:
        folder_name (str): Name of the folder to create
    """
    try:
        # Check if folder already exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully!")
        else:
            print(f"Folder '{folder_name}' already exists.")
    except Exception as e:
        print(f"Error creating folder: {e}")


if __name__ == "__main__":
    # Example usage
    folder_name = "new_folder"
    create_folder(folder_name)
