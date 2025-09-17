import os
from openai import OpenAI
from dotenv import load_dotenv
import time
from pathlib import Path

load_dotenv()

# deepseek
DEEPSEEK_API_KEY = os.getenv("deepseek_API_KEY")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMClient:
    _instances = {}

    @classmethod
    def get_client(cls, llm_type="openai"):
        if llm_type not in cls._instances:
            if llm_type == "openai":
                cls._instances[llm_type] = OpenAI(api_key=OPENAI_API_KEY)
            elif llm_type == "deepseek":
                cls._instances[llm_type] = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Bailian base_url
                )
            else:
                raise ValueError(f"Unknown llm_type: {llm_type}")
        return cls._instances[llm_type]

# ----------------- General Batch Function -----------------
def upload_file(file_path,llm_type="openai"):
    print(f"Uploading JSONL file with request information...")
    client = LLMClient.get_client(llm_type)
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    print(f"File uploaded successfully. File ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(input_file_id,llm_type="openai"):
    print(f"Creating batch job based on input file ID...")
    client = LLMClient.get_client(llm_type)
    batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Batch job created successfully. Batch ID: {batch.id}\n")
    return batch.id

def check_job_status(batch_id,llm_type="openai"):
    print(f"Checking batch job status...")
    client = LLMClient.get_client(llm_type)
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Batch job status: {batch.status}\n")
    return batch.status

def get_output_id(batch_id,llm_type="openai"):
    print(f"Retrieving output file ID from successful batch requests...")
    client = LLMClient.get_client(llm_type)
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Output file ID: {batch.output_file_id}\n")
    return batch.output_file_id

def get_error_id(batch_id,llm_type="openai"):
    print(f"Retrieving error file ID from failed batch requests...")
    client = LLMClient.get_client(llm_type)
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Error file ID: {batch.error_file_id}\n")
    return batch.error_file_id

def download_results(output_file_id, output_file_path,llm_type="openai"):
    print(f"Downloading batch job successful results...")
    client = LLMClient.get_client(llm_type)
    content = client.files.content(output_file_id)
    # Print part of the content for testing
    print(f"Preview of successful results (first 1000 chars):{content.text[:1000]}...\n")
    # Save the result file locally
    content.write_to_file(output_file_path)
    print(f"Full output results saved to result.jsonl\n")

def download_errors(error_file_id, error_file_path,llm_type="openai"):
    print(f"Downloading batch job error results...")
    client = LLMClient.get_client(llm_type)
    content = client.files.content(error_file_id)
    print(f"Preview of error results (first 1000 chars):{content.text[:1000]}...\n")
    content.write_to_file(error_file_path)
    print(f"Full error results saved to error.jsonl\n")

def batch_data(input_file_path,output_file_path,error_file_path,llm_type="openai"):
    try:
        # Step 1: Upload JSONL file
        input_file_id = upload_file(input_file_path)
        # Step 2: Create batch job
        batch_id = create_batch_job(input_file_id)
        # Step 3: Poll job status
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = check_job_status(batch_id)
            print(f"Waiting for job to complete...")
            time.sleep(10)
        # Step 4: Handle job status
        if status == "failed":
            client = LLMClient.get_client(llm_type)
            batch = client.batches.retrieve(batch_id)
            print(f"Batch job failed. Error details:{batch.errors}\n")
            return
        # Step 5: Download results or errors
        output_file_id = get_output_id(batch_id)
        if output_file_id:
            download_results(output_file_id, output_file_path)
        error_file_id = get_error_id(batch_id)
        if error_file_id:
            download_errors(error_file_id, error_file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    input_file = "../Leprosy/by_year/PMID_All.jsonl"
    output_file = "../Leprosy/by_year/PMID_All_output.jsonl"
    error_file = "../Leprosy/by_year/PMID_All_error.jsonl"

   # Use OpenAI
    batch_data(
        input_file_path=input_file,
        output_file_path=output_file,
        error_file_path=error_file,
    )
