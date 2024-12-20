import json
import glob
import os
import argparse
from pathlib import Path

def process_text_file(file_path):
    """Process a text file line by line, yielding cleaned lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Clean and strip the line
            line = line.strip()
            if line:  # Only yield non-empty lines
                yield line

def convert_txt_to_jsonl(input_dir, output_file):
    """
    Convert all .txt files in a directory to a single JSONL file,
    where each line from input becomes a separate JSON entry.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Get all .txt files
    txt_files = list(input_path.glob("**/*"))
    if not txt_files:
        raise ValueError(f"No files found in {input_dir}")
    
    print(f"Found {len(txt_files)} files")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    line_count = 0
    file_count = 0
    
    with output_path.open('w', encoding='utf-8') as f_out:
        for txt_file in txt_files:
            try:
                file_line_count = 0
                for line in process_text_file(txt_file):
                    # Create and write JSON entry for each line
                    json_obj = {"text": line}
                    f_out.write(json.dumps(json_obj) + '\n')
                    line_count += 1
                    file_line_count += 1
                
                file_count += 1
                if file_count % 10 == 0:
                    print(f"Processed {file_count}/{len(txt_files)} files... ({line_count} total lines)")
                
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")
    
    print(f"\nConversion complete:")
    print(f"- Files processed: {file_count}")
    print(f"- Total lines converted: {line_count}")
    print(f"- Output saved to: {output_file}")
    
    # Print a sample entry
    print("\nSample entries from output file:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Show first 3 entries
                break
            print(f"Entry {i+1}:", line.strip())

def main():
    parser = argparse.ArgumentParser(description='Convert text files to JSONL format for Pythia model training')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input .txt files')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to output JSONL file')
    
    args = parser.parse_args()
    
    try:
        convert_txt_to_jsonl(args.input_dir, args.output_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()