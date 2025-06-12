import json
import sys

def calculate_total_duration(file_path):
    # Initialize the total duration
    total_duration = 0.0

    # Open the JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object
            data = json.loads(line)
            # Add the duration to the total
            total_duration += data.get('duration', 0.0)

    # Print the total duration
    print(f"Total Duration: {total_duration/3600} hours")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_jsonl_file>")
    else:
        file_path = sys.argv[1]
        calculate_total_duration(file_path)