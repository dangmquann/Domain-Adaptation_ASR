import os

def check_and_create_source(file_path, new_content):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as file:
        file.write(new_content)

# Example usage
file_path = 'source.txt'
new_content = 'This is the new content for the file.'
check_and_create_source(file_path, new_content)