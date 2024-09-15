from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PyPDF2
import docx
import tiktoken
from io import BytesIO
import json
import time

app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Replace with your actual API key
api_key = "AIzaSyDSdQssjbwc-I-4aHQyi2MS-etlbKD92mY"

# Check if the API key is provided
if api_key == "YOUR_API_KEY_HERE":
    raise ValueError("Please replace 'YOUR_API_KEY_HERE' with your actual API key.")
else:
    genai.configure(api_key=api_key)

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

# Function to get response from Gemini API with retry logic
def get_gemini_response(input_text, context, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([context, input_text])
            
            # Log the raw response for debugging
            print(f"Raw response: {response.text.strip()}")
            
            if not response.text.strip():
                raise ValueError("Received an empty response from the API.")
            
            # Ensure response is in valid JSON format (removes code block markers if they exist)
            if response.text.startswith("```") and response.text.endswith("```"):
                clean_response = response.text.strip("```").strip("json")
                return clean_response.strip()
            
            return response.text.strip()
        except Exception as e:
            # Log the full exception message
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if "quota" in str(e).lower() or "500" in str(e):
                time.sleep(delay)
                attempt += 1
            else:
                raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    raise HTTPException(status_code=500, detail="We're unable to process your request right now. Please try again later or contact support.")

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

        context = ("Generate a detailed SOAP note and summaries based on the following psychotherapy transcript. "
                   "The response should be in JSON format with the following keys: 'subjective', 'objective', 'assessment', "
                   "'plan', 'clinicianSummary', 'clientSummary'. Please format the response exactly as follows, without any additional formatting or escaping.")
        soap_notes = get_gemini_response(transcript_text, context)

        try:
            # Try to parse the response as JSON
            soap_notes_json = json.loads(soap_notes)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a message with the raw response
            raise HTTPException(status_code=500, detail=f"Failed to parse SOAP notes. Response was: {soap_notes}")

        # Return the SOAP notes directly
        return JSONResponse(content=soap_notes_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SOAP notes: {str(e)}")
