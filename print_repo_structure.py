import os

def print_directory_structure(start_path='.', indent=0):
    """
    Recursively print the directory structure starting from start_path.
    indent parameter controls the indentation level for visual hierarchy.
    """
    # Get all entries in the directory
    entries = sorted(os.listdir(start_path))
    for entry in entries:
        # Skip hidden files and directories (starting with .)
        if entry.startswith('.'):
            continue
        path = os.path.join(start_path, entry)
        # Print the entry with appropriate indentation
        print('  ' * indent + '├── ' + entry)
        # If it's a directory, recurse into it
        if os.path.isdir(path):
            print_directory_structure(path, indent + 1)

def main():
    print("Repository Structure:")
    print_directory_structure()

if __name__ == "__main__":
    main()