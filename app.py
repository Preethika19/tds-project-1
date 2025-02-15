from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Dict
import phase_a as fun
import os
import json
import requests
from datetime import datetime, timedelta
import subprocess
import base64
import itertools
import numpy as np
import sqlite3
from pathlib import Path

# AI Proxy API settings
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Replace with your actual token

app = FastAPI()

# Define a Pydantic model to handle the request body


function_names = {
    "install_and_run_script" : fun.install_and_run_script,
    "format_markdown" : fun.format_markdown,
    "get_number_of_days" : fun.get_number_of_days,
    "sort_contacts" : fun.sort_contacts,
    "write_logs" : fun.write_logs,
    "extract_markdown_titles" : fun.extract_markdown_titles,
    "extract_email_sender" : fun.extract_email_sender,
    "extract_credit_card_detail" : fun.extract_credit_card_detail,
    "find_most_similar_comments" : fun.find_most_similar_comments,
    "calculate_total_sales_gold_tickets" : fun.calculate_total_sales_gold_tickets,
    "fetch_and_save_api_data" : fun.fetch_and_save_api_data,
    "convert_markdown_to_html" : fun.convert_markdown_to_html,
    "resize_and_compress_image" : fun.resize_and_compress_image,
    "transcribe_audio" : fun.transcribe_audio,
    "filter_csv_and_return_json" : fun.filter_csv_and_return_json,
    "scrape_website" : fun.scrape_website,
    "clone_and_commit_git_repo" : fun.clone_and_commit_git_repo,
    "run_sql_query" : fun.run_sql_query,
    "handle_file_delete" : fun.handle_file_delete
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

# Define the POST endpoint
@app.post("/run")
async def run_endpoint(task: str):
    user_prompt = task
    print(user_prompt)

    payload = {
        "model": "gpt-4o-mini",
        "functions": fun.function_descriptions,
        "function_call": "auto",
        "messages": [{"role": "user", "content": user_prompt}]
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for non-200 responses
        response_json = response.json()  # Parse JSON safely

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"AI Proxy Request Failed: {str(e)}")

    # Check if function call exists in response
    if "choices" in response_json and response_json["choices"]:
        function_call = response_json["choices"][0]["message"].get("function_call")
        
        if function_call:
            function_name = function_call.get("name")
            try:
                function_args = json.loads(function_call.get("arguments"))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON arguments from AI response")

            print("Function Name:", function_name)
            print("Arguments:", function_args)

            if function_name not in function_names:
                raise HTTPException(status_code=400, detail=f"Unknown function: {function_name}")

            try:
                result = function_names[function_name](**function_args)
                return result  # Success: 200 OK
            except HTTPException as e:
                raise e  # If function already raises an HTTPException, re-raise it
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Function execution error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No function call found in AI response")
    else:
        raise HTTPException(status_code=400, detail="Error: No valid choices found in AI response")

@app.get("/read")
async def read_file(path: str):
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return PlainTextResponse(content=file_path.read_text(encoding="utf-8"))

