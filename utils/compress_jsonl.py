import gzip
import sys

def compress_jsonl(input_file):
    output_file = input_file + ".gz"
    with open(input_file, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
        f_out.writelines(f_in)
    print(f"Compressed {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compress_jsonl.py <path_to_jsonl_file>")
    else:
        file_path = sys.argv[1]
        compress_jsonl(file_path)