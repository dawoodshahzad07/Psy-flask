from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI
import PyPDF2
import docx
import tiktoken  # For token counting
from io import BytesIO
import os
import json

app = FastAPI()

# Set up the OpenAI client
client = OpenAI(
    api_key="up_QnlPsfFqCqUDAfi3N68kMRgzDGjix",  # Replace with your OpenAI API key
    base_url="https://api.upstage.ai/v1/solar"
)

# List of psychotherapy-related keywords for context filtering
psychotherapy_keywords = ["therapy", "therapist", "session", "client", "counseling", "psychotherapy", "mental health"]

# Function to count tokens (requires tiktoken library)
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # Choose appropriate encoding
    tokens = encoding.encode(text)
    return len(tokens)

# Function to truncate text to a specified number of tokens
def truncate_text(prompt, text, max_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(encoding.encode(prompt))
    content_tokens = max_tokens - prompt_tokens
    tokens = encoding.encode(text)
    
    if len(tokens) > content_tokens:
        tokens = tokens[:content_tokens]
        text = encoding.decode(tokens)
    
    return text

# Function to handle streaming response from Upstage's solar-pro model
# Function to handle streaming response from Upstage's solar-pro model
def get_streamed_response(text):
    # Token limit for the model
    max_tokens = 3000

    # Define the prompt with JSON format instructions
    prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript. 
    The response should be in JSON format with the following keys: "Subjective", "Objective", "Assessment", "Plan", "ClinicianSummary", "Client Summary".

    Please format the response exactly as follows:

    {{
        "Subjective": "<Subjective content>",
        "Objective": "<Objective content>",
        "Assessment": "<Assessment content>",
        "Plan": "<Plan content>",
        "Clinician Summary": "<Clinician summary content>",
        "Client Summary": "<Client summary content>"
    }}

    Transcript:
    {text}
    """
    
    # Truncate the text to fit within the token limit, including the prompt
    truncated_text = truncate_text(prompt, text, max_tokens)

    # Modify the prompt to specifically request SOAP note format and summaries
    final_prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript. 
    The response should be in JSON format with the following keys: "Subjective", "Objective", "Assessment", "Plan", "Clinician Summary", "Client Summary".

    Please format the response exactly as follows:

    {{
        "Subjective": "<Subjective content>",
        "Objective": "<Objective content>",
        "Assessment": "<Assessment content>",
        "Plan": "<Plan content>",
        "Clinician Summary": "<Clinician summary content>",
        "Client Summary": "<Client summary content>"
    }}

    Transcript:
    {truncated_text}
    """
    
    # Stream the API response as the output generates
    stream = client.chat.completions.create(
        model="solar-pro",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant. Please provide responses in JSON format as described."
        }, {
            "role": "user",
            "content": final_prompt
        }],
        stream=True,
    )

    # Yield chunks of streamed content
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    
    # Attempt to parse the response as JSON
    try:
        parsed_response = json.loads(full_response)
    except json.JSONDecodeError:
        # If parsing fails, treat it as plain text
        return {"error": "Failed to parse response as JSON", "response": full_response}

    return parsed_response


# Helper function to extract text from file-like objects (PDF, DOCX, or TXT)
def extract_text_from_file(file: BytesIO, file_type: str):
    if file_type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    elif file_type == "text/plain":
        return file.read().decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

# Function to check if the transcript is related to psychotherapy
def is_relevant_psychotherapy_text(text):
    return any(keyword.lower() in text.lower() for keyword in psychotherapy_keywords)

# FastAPI Endpoint to upload a file and generate SOAP notes
@app.post("/generate-soap/")
async def generate_soap(file: UploadFile = File(...)):
    try:
        # Extract text from the uploaded file
        file_content = await file.read()
        transcript_text = extract_text_from_file(BytesIO(file_content), file.content_type)

        if not transcript_text:
            raise HTTPException(status_code=400, detail="No text extracted from the file")

        # Check if the transcript is related to psychotherapy
        if not is_relevant_psychotherapy_text(transcript_text):
            raise HTTPException(status_code=400, detail="The transcript does not appear to be related to psychotherapy")

        # Generate SOAP notes using the extracted text
        soap_notes = get_streamed_response(transcript_text)

        # Return response based on its format
        if "error" in soap_notes:
            return PlainTextResponse(content=soap_notes["response"], status_code=200)
        return JSONResponse(content={"soap_notes": soap_notes})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SOAP notes: {str(e)}")
