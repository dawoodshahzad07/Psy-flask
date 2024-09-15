from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import PyPDF2
import docx
import tiktoken
from io import BytesIO
import json

app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Set up the OpenAI client
client = OpenAI(
    api_key="up_QnlPsfFqCqUDAfi3N68kMRgzDGjix",  # Replace with your OpenAI API key
    base_url="https://api.upstage.ai/v1/solar"
)

# List of psychotherapy-related keywords for context filtering
psychotherapy_keywords = ["therapy", "therapist", "session", "client", "counseling", "psychotherapy", "mental health"]

# Function to count tokens (requires tiktoken library)
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
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
def get_streamed_response(text):
    max_tokens = 3000

    prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript. 
    The response should be in JSON format with the following keys: "subjective", "objective", "assessment", "plan", "clinicianSummary", "clientSummary".

    Please format the response exactly as follows, without any additional formatting or escaping:

    {{
        "subjective": "<Subjective content>",
        "objective": "<Objective content>",
        "assessment": "<Assessment content>",
        "plan": "<Plan content>",
        "clinicianSummary": "<Clinician summary content>",
        "clientSummary": "<Client summary content>"
    }}

    Transcript:
    {text}
    """
    
    truncated_text = truncate_text(prompt, text, max_tokens)

    final_prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript. 
    The response should be in JSON format with the following keys: "subjective", "objective", "assessment", "plan", "clinicianSummary", "clientSummary".

    Please format the response exactly as follows, without any additional formatting or escaping:

    {{
        "subjective": "<Subjective content>",
        "objective": "<Objective content>",
        "assessment": "<Assessment content>",
        "plan": "<Plan content>",
        "clinicianSummary": "<Clinician summary content>",
        "clientSummary": "<Client summary content>"
    }}

    Transcript:
    {truncated_text}
    """
    
    stream = client.chat.completions.create(
        model="solar-pro",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant. Please provide responses in JSON format as described, without any additional formatting or escaping."
        }, {
            "role": "user",
            "content": final_prompt
        }],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    
    def clean_and_truncate(text, max_length=1000):
        # Remove newlines and extra spaces
        cleaned = ' '.join(text.split())
        # Truncate if necessary
        return (cleaned[:max_length] + '...') if len(cleaned) > max_length else cleaned

    try:
        # First, try to parse the entire response as JSON
        parsed_response = json.loads(full_response)
    except json.JSONDecodeError:
        # If that fails, try to extract a JSON object from the response
        try:
            start = full_response.find('{')
            end = full_response.rfind('}') + 1
            if start != -1 and end != -1:
                parsed_response = json.loads(full_response[start:end])
            else:
                raise json.JSONDecodeError("No valid JSON found", full_response, 0)
        except json.JSONDecodeError:
            # If JSON extraction fails, create a structured response with cleaned content
            return {
                "subjective": clean_and_truncate(full_response),
                "objective": "Error: Could not parse response",
                "assessment": "Error: Could not parse response",
                "plan": "Error: Could not parse response",
                "clinicianSummary": "Error: Could not parse response",
                "clientSummary": "Error: Could not parse response"
            }

    # Ensure all required keys are present and clean/truncate the content
    required_keys = ["subjective", "objective", "assessment", "plan", "clinicianSummary", "clientSummary"]
    for key in required_keys:
        if key not in parsed_response or not isinstance(parsed_response[key], str):
            parsed_response[key] = "Not provided"
        else:
            parsed_response[key] = clean_and_truncate(parsed_response[key])

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
        file_content = await file.read()
        transcript_text = extract_text_from_file(BytesIO(file_content), file.content_type)

        if not transcript_text:
            raise HTTPException(status_code=400, detail="No text extracted from the file")

        if not is_relevant_psychotherapy_text(transcript_text):
            raise HTTPException(status_code=400, detail="The transcript does not appear to be related to psychotherapy")

        soap_notes = get_streamed_response(transcript_text)

        # Return the SOAP notes directly, without the "soapNotes" wrapper
        return JSONResponse(content=soap_notes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SOAP notes: {str(e)}")
