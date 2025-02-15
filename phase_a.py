import os
import json
import requests
from datetime import datetime, timedelta
import subprocess
import base64
import itertools
import numpy as np
import sqlite3
import re
import sys
from dateutil.parser import parse
from fastapi import HTTPException
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
import markdown
from pydub import AudioSegment
import speech_recognition as sr
import shutil


# AI Proxy API settings
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Replace with your actual token
# print(AIPROXY_TOKEN)

function_descriptions = [
    {
        "name": "install_and_run_script",
        "description": "Installs required packages if necessary and runs a script from a given URL with specified arguments.",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of packages to install before running the script."
                },
                "script_url": {
                    "type": "string",
                    "description": "The URL of the Python script to download and run."
                },
                "arguments": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of arguments to pass to the script."
                }
            },
            "required": ["packages", "script_url", "arguments"]
        }
    },
    {
        "name": "format_markdown",
        "descriptions": "Format given markdown file using prettier",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Markdown file name",
                },
                "version": {
                    "type": "string",
                    "description": "Prettier version",
                },
            },
            "required": ["input_file", "version"],
        },
    },
    {
        "name": "get_number_of_days",
        "descriptions": "Count the number of days in the given list",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {
                    "type": "string",
                    "description": "The day to be extracted",
                },
                "source_file": {
                    "type": "string",
                    "description": "The file with list of dates",
                },
                "destination_file": {
                    "type": "string",
                    "description": "The file where count is written",
                },
            },
            "required": ["day", "source_file", "destination_file"],
        },
    },
    {
        "name": "sort_contacts",
        "description": "Sorts an array of contacts based on specified sorting keys and writes the sorted list to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to the input JSON file containing an array of contacts."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output JSON file where the sorted contacts will be written."
                },
                "sort_keys": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of keys to sort by, in priority order. Example: ['last_name', 'first_name']."
                },
            },
            "required": ["input_file", "output_file", "sort_keys"]
        }
    },
    {
        "name": "write_logs",
        "description": "Write the given number of lines from the specified files and write them to given output file in specified order",
        "parameters": {
            "type": "object",
            "properties": {
                "file_type": {
                    "type": "string",
                    "description": "Type of files to be read."
                },
                "directory": {
                    "type": "string",
                    "description": "The directory from which files should be read."
                },
                "output_file": {
                    "type": "string",
                    "description": "File in which output should be written."
                },
                "num_files": {
                    "type": "integer",
                    "description": "Number of files to be read."
                },
                "files_order": {
                    "type": "string",
                    "description": "Order in which files in directory should be read."
                },
                "num_lines": {
                    "type": "integer",
                    "description": "Number of lines to be read from each file."
                },
                "sort_order": {
                    "type": "string",
                    "description": "Order in which output should be written."
                },
            },
            "required": ["file_type", "directory", "output_file", "num_files", "files_order", "num_lines", "sort_order"]
        }
    },
    {
        "name": "extract_markdown_titles",
        "description": "Extracts the first occurrence of each H1 title from Markdown files in a directory and creates an index file that maps each file to its title.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory containing the files."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output JSON file where the output will be saved."
                },
                "file_extension": {
                    "type": "string",
                    "default": ".md",
                    "description": "File extension/type."
                },
                "h1_indicator": {
                    "type": "string",
                    "default": "#",
                    "description": "The character(s) indicating an H1 in Markdown."
                },
                "strip_prefix": {
                    "type": "string",
                    "description": "Prefix to be stripped from filenames in the output."
                }
            },
            "required": ["directory", "output_file"]
        }
    },
    {
        "name": "extract_email_sender",
        "description": "Extracts the sender's email address from an email message and writes it to a new file.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to the input file containing the email message."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output file."
                },
            },
            "required": ["input_file", "output_file"]
        }
    },
    {
        "name": "extract_credit_card_detail",
        "description": "Extracts given field from an image and writes it to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to the input image file."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output file."
                },
                "extract_item": {
                    "type": "string",
                    "description": "What to extract from image."
                },

            },
            "required": ["input_file", "output_file", "extract_item"]
        }
    },
    {
        "name": "find_most_similar_comments",
        "description": "Finds the most similar pair of comments using embeddings and writes them to an output file, one per line.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to the input file containing the list of comments."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output file where the most similar pair of comments will be written."
                },
                "embedding_model": {
                    "type": "string",
                    "description": "The model or method used to generate the embeddings for the comments."
                },
            },
            "required": ["input_file", "output_file",]
        }
    },
    {
        "name": "calculate_total_sales_gold_tickets",
        "description": "Calculates the total sales of all items for the given ticket type and writes the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "database_file": {
                    "type": "string",
                    "description": "Path to the SQLite database file containing the ticket sales."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to the output file where the total sales amount will be written."
                },
                "ticket_type": {
                    "type": "string",
                    "description": "The ticket type to calculate sales for (default is 'Gold')."
                }
            },
            "required": ["database_file", "ticket_type", "output_file"]
        }
    },
        {
        "name": "handle_file_delete",
        "description": "Recognise delete,remove,move,renmae requests",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path of the file to operate on."
                },
            },
            "required":  ["file_path"]
        }
    },
    {
        "name": "fetch_and_save_api_data",
        "description": "Fetches data from a given API URL and saves it to a specified file.",
        "parameters": {
            "type": "object",
            "properties": {
                "api_url": {
                    "type": "string",
                    "description": "The API URL to fetch data from."
                },
                "output_file": {
                    "type": "string",
                    "description": "The file path where the fetched data should be saved."
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers to include in the request.",
                    "additionalProperties": {
                        "type": "string"
                    }
                }
            },
            "required": ["api_url", "output_file"]
        }
    },
    {
        "name": "clone_and_commit_git_repo",
        "description": "Clones a Git repository and makes a commit.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "URL of the Git repository."
                },
                "local_dir": {
                    "type": "string",
                    "description": "Path where the repository should be cloned."
                },
                "commit_message": {
                    "type": "string",
                    "description": "Commit message for the change."
                }
            },
            "required": ["repo_url", "local_dir", "commit_message"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Runs a SQL query on a SQLite or DuckDB database.",
        "parameters": {
            "type": "object",
            "properties": {
                "database_file": {
                    "type": "string",
                    "description": "Path to the SQLite or DuckDB database file."
                },
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute."
                }
            },
            "required": ["database_file", "query"]
        }
    },
    {
        "name": "scrape_website",
        "description": "Extracts data from a website.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to scrape."
                },
                "element_selector": {
                    "type": "string",
                    "description": "CSS selector for the elements to extract."
                },
                "output_file": {
                    "type": "string",
                    "description": "The file path where the fetched data should be saved."
                }
            },
            "required": ["url", "element_selector"]
        }
    },
    {
        "name": "resize_and_compress_image",
        "description": "Resizes and compresses an image to fit within a specified file size limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image."
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the compressed image."
                },
                "max_file_size_kb": {
                    "type": "integer",
                    "description": "Maximum allowed file size in KB."
                },
                "width": {
                    "type": "integer",
                    "description": "New width of the image (optional, default keeps aspect ratio)."
                },
                "height": {
                    "type": "integer",
                    "description": "New height of the image (optional, default keeps aspect ratio)."
                }
            },
            "required": ["image_path", "max_file_size_kb"]
        }
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribes audio from an MP3 file.",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_file": {
                    "type": "string",
                    "description": "Path to the MP3 audio file."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the result."
                }
            },
            "required": ["audio_file"]
        }
    },
    {
        "name": "convert_markdown_to_html",
        "description": "Converts Markdown content to HTML.",
        "parameters": {
            "type": "object",
            "properties": {
                "markdown_file": {
                    "type": "string",
                    "description": "Path to the Markdown file."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the HTML file."
                }
            },
            "required": ["markdown_file", "output_file"]
        }
    },
    {
        "name": "filter_csv_and_return_json",
        "description": "Filters a CSV file based on given criteria and returns JSON data.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv_file": {
                    "type": "string",
                    "description": "Path to the CSV file."
                },
                "filter_column": {
                    "type": "string",
                    "description": "The column name to filter by."
                },
                "filter_value": {
                    "type": "string",
                    "description": "The value to filter for."
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the output file."
                }
            },
            "required": ["csv_file", "filter_column", "filter_value"]
        }
    },
]


def install_and_run_script(packages: list, script_url: str, arguments: list):
    # Step 1: Install required packages
    for package in packages:
        try:
            __import__(package)
            print(f"'{package}' is already installed.")
        except ImportError:
            print(f"Installing '{package}'...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"Failed to install package '{package}': {str(e)}")

    # Step 2: Download the script
    try:
        response = requests.get(script_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download script: {str(e)}")

    script_path = "datagen.py"
    try:
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(response.text)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write script file: {str(e)}")

    # Step 3: Run the script
    print(f"Running the script {script_path} with arguments: {arguments}")
    try:
        result = subprocess.run(
            [sys.executable, script_path] + arguments,
            capture_output=True,
            text=True,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=400, detail="Script execution timed out.")
    except subprocess.SubprocessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {str(e)}")

    if result.returncode == 0:
        return {"status": "success", "output": result.stdout}
    else:
        raise HTTPException(status_code=400, detail=f"Script error: {result.stderr}")


def format_markdown(input_file: str, version: str) -> str:
    # Check if the input file is inside the /data directory
    if not input_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Error: The file must be inside the /data directory.")

    # Check if the file exists before proceeding
    if not os.path.isfile(input_file):
        raise HTTPException(status_code=404, detail="Error: File not found.")

    try:
        # Install the specified version of Prettier if necessary
        subprocess.run(['npm', 'install', f'prettier@{version}'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error installing Prettier: {e.stderr.decode()}")

    try:
        # Run Prettier to format the input file
        subprocess.run(['npx', 'prettier', '--write', input_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error formatting file: {e.stderr.decode()}")

    return {"status": "success", "message": "Markdown file formatted successfully."}


def get_number_of_days(day: str, source_file: str, destination_file: str) -> str:
    # Validate that both files are inside the /data directory
    if not source_file.startswith("/data/") or not destination_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail= "Both input and output files must be inside the /data directory.")

    # Check if the source file exists
    if not os.path.isfile(source_file):
        raise HTTPException(status_code=404, detail="Source file not found.")

    try:
        # Read the dates from the input file
        with open(source_file, 'r', encoding="utf-8") as file:
            dates = file.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading source file: {str(e)}")

    try:
        # Count occurrences of the specified weekday
        weekday_count = sum(1 for date in dates if parse(date.strip()).strftime("%A").lower() == day.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing dates: {str(e)}")

    try:
        # Write the count to the output file
        with open(destination_file, 'w', encoding="utf-8") as file:
            file.write(str(weekday_count))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

    return {"status": "success", "message": f"{day}s counted: {weekday_count}"}


def sort_contacts(input_file: str, output_file: str, sort_keys: list) -> str:
    # Validate that both files are inside the /data directory
    if not input_file.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Check if the input file exists
    if not os.path.isfile(input_file):
        raise HTTPException(status_code=404, detail="Input file not found.")

    try:
        # Read the contacts from the input file
        with open(input_file, 'r', encoding="utf-8") as file:
            contacts = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in input file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading input file: {str(e)}")

    # Validate that contacts is a list
    if not isinstance(contacts, list):
        raise HTTPException(status_code=400, detail="Invalid data: Expected a list of contacts.")

    try:
        # Sort contacts based on the provided sort keys
        sorted_contacts = sorted(contacts, key=lambda x: tuple(x.get(key, "") for key in sort_keys))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sort key: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sorting contacts: {str(e)}")

    try:
        # Write the sorted contacts to the output file
        with open(output_file, 'w', encoding="utf-8") as file:
            json.dump(sorted_contacts, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

    return {"status": "success", "message": "Contacts sorted successfully."}


'''
def write_logs(file_type: str, directory: str, output_file: str, num_files: int,
               files_order: str, num_lines: int, sort_order: str) -> str:

    if not os.path.commonpath([directory, '/data']) == '/data' or not os.path.commonpath([output_file, '/data']) == '/data':
        return "Error: Both input and output files must be inside the /data directory."

    # Ensure the directory exists and contains the given file type
    if not os.path.exists(directory):
        return f"Error: Directory '{directory}' does not exist."

    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith(file_type)]

    files = sorted(files, key=os.path.getmtime, reverse=(sort_order == 'desc'))

    # Limit the number of files to the requested number
    selected_files = files[:num_files]

    # Write the specified number of lines from each file to the output file
    with open(output_file, 'w') as output:
        for file in selected_files:
            with open(file, 'r') as f:
                # Read the specified number of lines from each file
                for _ in range(num_lines):
                    line = f.readline()
                    if not line:
                        break  # Stop if fewer lines are available
                    output.write(line)

    return f"{num_files} files processed, {num_lines} lines written to {output_file}."

'''

def write_logs(file_type: str, directory: str, output_file: str, num_files: int,
               files_order: str, num_lines: int, sort_order: str) -> str:
               # Ensure input and output paths are within /data
    if not os.path.abspath(directory).startswith('/data') or not os.path.abspath(output_file).startswith('/data'):
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Ensure the directory exists
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail=f"Directory '{directory}' does not exist or is not a directory.")

    # Get all files of the given type
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_type)]

    if not files:
        raise HTTPException(status_code=404, detail=f"No files of type '{file_type}' found in directory '{directory}'.")

    # Sort files based on modification time
    reverse_sort = (files_order == 'desc')  # Newest first if 'desc', oldest first if 'asc'
    files = sorted(files, key=os.path.getmtime, reverse=reverse_sort)

    # Limit number of files to process
    selected_files = files[:min(num_files, len(files))]

    if not selected_files:
        raise HTTPException(status_code=400, detail="No files selected based on the input parameters.")

    try:
        # Write the specified number of lines from each file to output
        with open(output_file, 'w', encoding="utf-8") as output:
            for file in selected_files:
                with open(file, 'r', encoding="utf-8") as f:
                    for _ in range(num_lines):
                        line = f.readline()
                        if not line:
                            break  # Stop if fewer lines are available
                        output.write(line)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

    return {"status": "success", "message": f"{len(selected_files)} files processed, {num_lines} lines written to {output_file}."}


def extract_markdown_titles(directory: str, output_file: str, file_extension: str = '.md',
                            h1_indicator: str = '#', strip_prefix: str = '') -> str:

    # Validate input and output paths to ensure they're inside the /data directory
    if not os.path.commonpath([directory, '/data']) == '/data' or not os.path.commonpath([output_file, '/data']) == '/data':
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Ensure the directory exists
    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail=f"Directory '{directory}' does not exist.")

    # Get all markdown files in the directory with the specified extension
    md_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(file_extension):  # Check for .md extension
                md_files.append(os.path.join(root, f))  # Append full path

    if not md_files:
        raise HTTPException(status_code=404, detail="No markdown files found in the specified directory.")

    # Initialize the dictionary to store file-title mappings
    file_title_map = {}

    # Process each markdown file
    for file in md_files:
        try:
            with open(file, 'r', encoding="utf-8") as f:
                for line in f:
                    # Check for an H1 indicator (line starting with '#')
                    if line.startswith(h1_indicator):
                        # Remove the H1 indicator and any leading/trailing spaces
                        title = line[len(h1_indicator):].strip()
                        file_title_map[file.replace(directory, '')] = title
                        break  # Stop after the first occurrence of an H1 title
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file {file}: {str(e)}")

    try:
        # Write the result to the output JSON file
        with open(output_file, 'w', encoding="utf-8") as output:
            json.dump(file_title_map, output, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

    return {"status": "success", "message": f"Markdown titles extracted and saved to '{output_file}'."}

def extract_email_sender(input_file: str, output_file: str) -> str:
    # Ensure files are inside /data
    if not input_file.startswith("/data") or not output_file.startswith("/data"):
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Read the email content
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            email_content = file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading input file: {str(e)}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email address from the following email and return only the email address."},
            {"role": "user", "content": email_content}
        ]
    }

    # Send request to LLM
    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=data)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending request to LLM: {str(e)}")

    if response.status_code == 200:
        try:
            sender_email = response.json()['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=500, detail="Error parsing the response from LLM.")
        
        # Save the extracted email to output file
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(sender_email)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

        return {"status": "success", "message": f"Sender's email extracted and saved to {output_file}."}
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Error: {response.status_code} - {response.text}")


def extract_credit_card_detail(input_file: str, output_file: str, extract_item: str) -> str:
    # Ensure files are inside /data
    if not os.path.commonpath([input_file, '/data']) == '/data' or not os.path.commonpath([output_file, '/data']) == '/data':
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    extract_item = re.sub(r'(?i)credit card', '', extract_item).strip()

    # Read and encode image in base64
    try:
        with open(input_file, "rb") as file:
            image_data = base64.b64encode(file.read()).decode("utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"The file {input_file} does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    # Format base64 image as a data URL
    image_url = f"data:image/png;base64,{image_data}"

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "perform ocr on this image"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for non-200 responses
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending request: {str(e)}")

    # Process response
    try:
        print(response.json())
        image_details = response.json()['choices'][0]['message']['content'].strip().replace(" ", "")
        data_ = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"separate only {extract_item} from {image_details} and give that field only. Dont give any header or additional sentence."}
                    ]
                }
            ]
        }

        response = requests.post(AIPROXY_URL, headers=headers, json=data_)
        response.raise_for_status()  # Raise an error for non-200 responses
        print(response.json())
        item_ = response.json()['choices'][0]['message']['content'].strip().replace(" ", "")

        # Save the extracted credit card number to output file
        try:
            with open(output_file, "w") as file:
                file.write(item_)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

        return {"status": "success", "message": f"Credit card number ITEM saved to {output_file}."}
    except KeyError:
        raise HTTPException(status_code=500, detail=f"Error: Unexpected response format - {response.json()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")

def find_most_similar_comments(input_file: str, output_file: str) -> str:
    # Ensure both files are inside /data
    if not os.path.commonpath([input_file, '/data']) == '/data' or not os.path.commonpath([output_file, '/data']) == '/data':
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Read comments from file
    try:
        with open(input_file, "r") as file:
            comments = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Error: The file {input_file} does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Error: Not enough comments to find a similar pair.")

    comments = [comment.replace("'", "\"") for comment in comments]

    # Prepare headers and payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "User-Agent": "Mozilla/5.0 (compatible; MyPythonClient/1.0)"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": comments
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for non-200 responses
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending request: {str(e)}")

    try:
        embeddings = np.array([item["embedding"] for item in response.json()["data"]])
    except KeyError:
        raise HTTPException(status_code=500, detail="Error: Unexpected response format - missing 'embedding' data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")

    # Calculate the most similar comments (for example, cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Find the most similar pair
    most_similar_pair = None
    max_similarity = -1
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                most_similar_pair = (comments[i], comments[j])

    # If similar comments are found, write them to the output file
    if most_similar_pair:
        try:
            with open(output_file, "w") as file:
                file.write(f"Most Similar Comments:\n{most_similar_pair[0]}\n{most_similar_pair[1]}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

        return {"status": "success", "message": f"Most similar comments saved to {output_file}."}
    else:
        raise HTTPException(status_code=404, detail="Error: No similar comments found.")

def calculate_total_sales_gold_tickets(database_file: str, output_file: str, ticket_type: str = "Gold") -> str:
    # Ensure files are inside /data
    if os.path.commonpath([database_file, '/data']) != '/data' or os.path.commonpath([output_file, '/data']) != '/data':
        raise HTTPException(status_code=400, detail="Both input and output files must be inside the /data directory.")

    # Connect to SQLite database
    try:
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()

        # Query to calculate total sales
        cursor.execute(
            "SELECT SUM(units * price) FROM tickets WHERE type = ?", (ticket_type,))
        total_sales = cursor.fetchone()[0]

        # Ensure total_sales is not None (if no matching records)
        total_sales = total_sales if total_sales else 0

        # Write the result to the output file
        try:
            with open(output_file, "w") as file:
                file.write(str(total_sales))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error writing to output file: {str(e)}")

        return f"Total sales for {ticket_type} tickets written to {output_file}."

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Error: The file {database_file} does not exist.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        if conn:
            conn.close()


def handle_file_delete(file_path: str):
    if not file_path.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    try:
        # Check if the input file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail=f"Markdown file '{file_path}' not found.")
        # File exists, but deletion is not allowed
        raise HTTPException(
            status_code=400, detail=f"Deletion is not allowed for '{file_path}'."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"An error occurred: {str(e)}"
        )

    

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def convert_ogg_to_wav(input_file: str, output_file: str):
    # Load OGG file

    audio = AudioSegment.from_ogg(input_file)
    # Export as WAV
    audio.export(output_file, format="wav")


def transcribe_audio(file_path: str, output_file: str = "output.txt") -> str:
    if not file_path.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    wav_file = file_path[:-4] + ".wav"
    if file_path.endswith(".mp3"):
        convert_mp3_to_wav(file_path, wav_file)
    elif file_path.endswith(".ogg"):
        convert_ogg_to_wav(file_path, wav_file)
    else:
        raise HTTPException(status_code=444, detail="Unsupported Media Type")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)

    try:
        transcription = recognizer.recognize_sphinx(audio_data)
        # Write the transcription to the output file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcription)

        return transcription
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech not recognized")
    except sr.RequestError:
        raise HTTPException(
            status_code=400, detail="Speech recognition service unavailable")


def convert_markdown_to_html(markdown_file: str, output_file: str):
    if not markdown_file.startswith("/data/") or not output_file.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    try:
        # Check if the input file exists
        if not os.path.exists(markdown_file):
            raise HTTPException(
                status_code=404, detail=f"Markdown file '{markdown_file}' not found.")

        # Read the Markdown content
        with open(markdown_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML
        html_content = markdown.markdown(md_content)

        # Write the HTML to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return f"HTML saved to {output_file}"

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Markdown file '{markdown_file}' not found.")

    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Permission denied: Unable to write to '{output_file}'.")

    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"File handling error: {str(e)}")


def resize_and_compress_image(image_path: str, max_file_size_kb: int, output_path: str = None, width: int = None, height: int = None):
    if not output_path.startswith("/data/") or not image_path.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    if output_path is None:
        output_path = image_path
    try:
        # Open the image
        img = Image.open(image_path)

        # Resize while maintaining aspect ratio if width/height provided
        if width and height:
            img = img.resize((width, height), Image.LANCZOS)
        elif width:
            aspect_ratio = img.height / img.width
            img = img.resize((width, int(width * aspect_ratio)), Image.LANCZOS)
        elif height:
            aspect_ratio = img.width / img.height
            img = img.resize(
                (int(height * aspect_ratio), height), Image.LANCZOS)

        # Start compression loop
        quality = 95  # Start with high quality
        img.save(output_path, format="JPEG", quality=quality)

        while os.path.getsize(output_path) > max_file_size_kb * 1024 and quality > 10:
            quality -= 5  # Reduce quality gradually
            img.save(output_path, format="JPEG", quality=quality)

        final_size = os.path.getsize(output_path) / 1024
        return f"Image saved at {output_path} with size {final_size:.2f} KB"

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Image file '{image_path}' not found.")

    except PermissionError:
        raise HTTPException(
            status_code=400, detail=f"Permission denied: Unable to write to '{output_path}'.")

    except OSError as e:
        raise HTTPException(
            status_code=400, detail=f"Image processing error: {str(e)}")

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Unexpected error: {str(e)}")


def filter_csv_and_return_json(csv_file: str, filter_column: str, filter_value: str, output_file: str):
    if not output_file.startswith("/data/") or not csv_file.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    try:
        df = pd.read_csv(csv_file)

        if filter_column not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{filter_column}' not found in CSV.")

        filtered_df = df[df[filter_column] == filter_value]

        if filtered_df.empty:
            raise HTTPException(
                status_code=404, detail="No matching records found.")

        json_data = filtered_df.to_json(orient="records")

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(json_data)

        return json_data

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"CSV file '{csv_file}' not found.")

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing CSV file.")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}")


def scrape_website(url: str, element_selector: str, output_file: str = "output.txt"):
    if not output_file.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTP error for non-200 responses
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Request failed: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.select(element_selector)

    if not elements:
        raise HTTPException(
            status_code=404, detail="No elements found with the given selector")

    element_texts = [element.text.strip() for element in elements]
    # Write the output to the specified file
    with open(output_file, "w", encoding="utf-8") as file:
        for text in element_texts:
            file.write(text + "\n")

    return element_texts


def clone_and_commit_git_repo(repo_url: str, local_dir: str, commit_message: str):
    if not local_dir.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    if not shutil.which("git"):
        raise HTTPException(
            status_code=400, detail="Git is not installed on the system")

    try:
        if not os.path.exists(local_dir):
            subprocess.run(["git", "clone", repo_url, local_dir], check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=400, detail="Invalid or unreachable repository URL")

    try:
        os.chdir(local_dir)
        subprocess.run(["git", "add", "."], check=True)
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message], capture_output=True, text=True)

        # Handle empty commit case
        if "nothing to commit" in commit_result.stdout.lower():
            return "No new changes to commit."

        return "Commit successful"

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400, detail=f"Git command failed: {e.stderr}")


def fetch_and_save_api_data(api_url: str, output_file: str, headers: dict = None):
    if not output_file.startswith("/data/"):
        raise HTTPException(
            status_code=400, detail="Error: The file must be inside the /data directory.")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for 4xx/5xx responses

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)

        return f"Data saved to {output_file}"

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"API request failed: {str(e)}")


def run_sql_query(database_file: str, query: str):
    # Check if the database file exists
    if not os.path.isfile(database_file):
        raise HTTPException(status_code=404)

    try:
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        conn.commit()
        return result

    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {str(e)}")

    except sqlite3.DatabaseError as e:
        raise HTTPException(
            status_code=400, detail=f"Database error: {str(e)}")

    finally:
        if conn:
            conn.close()

# user_prompt = "Format the contents of /data/project/format.md using prettier@3.3.2, updating the file in-place"
# user_prompt = "Write #of thursdays in /data/data.txt to the file /data/outputs/num-of-thurs.txt"
# user_prompt = "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and then by age,and write the result to /data/contacts-sorted.json"
# user_prompt = "Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first"
# user_prompt = ""
