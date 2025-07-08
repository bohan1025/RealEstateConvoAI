from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import whisper
import openai
import json
import pyodbc
from gtts import gTTS
import os
import logging
from typing import Dict, Any, List, Optional
import tempfile
import re
import librosa
import numpy as np
import openai
from scipy import signal
from contextlib import contextmanager
import threading
import uuid
from datetime import datetime
import requests
from openai import AzureOpenAI

# ElevenLabs configuration
ELEVENLABS_API_KEY = "sk_bd836f8e0a8e874a0ab8891a0724e835a67bd14155ef6ac8" 
ELEVENLABS_VOICE_ID = "ErXwobaYiN019PkySvjV"  # voice ID 

from elevenlabs import ElevenLabs

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

import warnings
import multiprocessing
import subprocess
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Voice Robot API", description="Complete voice interaction system")

# Connect to the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None

# Azure open AI information
# Old configuration - keeping for reference
# openai.api_type = "azure"
# openai.api_key = "mSSedelOpALeOCyKP4ssEipRpAkgAZz3v1kTIHBSGnrJqprIo349JQQJ99BGACL93NaXJ3w3AAABACOGf4ui" # API key
# openai.api_base = "https://aivoicetest.openai.azure.com/" # API end point
# openai.api_version = "2024-02-15-preview"

# New Azure OpenAI client initialization
openai_client = AzureOpenAI(
    api_key="mSSedelOpALeOCyKP4ssEipRpAkgAZz3v1kTIHBSGnrJqprIo349JQQJ99BGACL93NaXJ3w3AAABACOGf4ui",
    api_version="2024-02-15-preview",
    azure_endpoint="https://aivoicetest.openai.azure.com/"
)


# Azure SQL information
AZURE_SQL_CONFIG = {
    "driver": "ODBC Driver 18 for SQL Server",
    "server": "voiceai-sql-server.database.windows.net",
    "database": "RealEstateConvoAI",
    "username": "lorraine", # Azure database username
    "password": "Test123!Temp" # Azure database password
}

# Azure Voice AI information


# Azure Speech Service information
AZURE_SPEECH_CONFIG = {
    "subscription": "mSSedelOpALeOCyKP4ssEipRpAkgAZz3v1kTIHBSGnrJqprIo349JQQJ99BGACL93NaXJ3w3AAABACOGf4ui",  
    "region": "australiaeast"    
}

# ==================== Step 1: Recieve the front end audio file ====================
def save_uploaded_audio(file: UploadFile) -> str:
    """Save the uploaded audio file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    temp_path = temp_file.name
    temp_file.close()
    
    with open(temp_path, "wb") as f:
        content = file.file.read()
        f.write(content)
    
    logger.info(f"Audio saved: {temp_path}")
    # check whether the backend revcieve the audio file
    file_size = len(content)
    logger.info(f"Audio saved: {temp_path}, size: {file_size} bytes")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    
    # File size check
    if file_size < 2000:  # At least 2 seconds
        raise HTTPException(status_code=400, detail="Audio file too short. Please speak for at least 2 seconds.")
    
    if file_size < 5000:  
        logger.warning(f"Audio file seems small: {file_size} bytes")
    
    return temp_path

# As for Chrome browser, it will record the audio in webm format, so we need to convert it to wav format for ElevenLabs
def convert_webm_to_wav(webm_path: str) -> str:
    """Convert webm audio to wav format for ElevenLabs"""
    wav_path = webm_path.replace('.webm', '.wav')
    
    try:
        # use ffmpeg to convert
        cmd = [
            'ffmpeg', '-i', webm_path,
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ac', '1',               # single channel
            '-ar', '16000',           # 16kHz sampling rate
            '-y',                     # overwrite output file
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted {webm_path} to {wav_path}")
            return wav_path
        else:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            return webm_path  # return the original file
            
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        return webm_path  # return the original file

# ==================== Step 2: Speech to text -> ElevenLabs ====================
# Using 11labs as the primary STT tools
# Not working
def elevenlabs_stt(audio_path, api_key=ELEVENLABS_API_KEY):
    """Speech to text using ElevenLabs STT"""
    try:
        if audio_path.endswith('.webm'):
            wav_path = convert_webm_to_wav(audio_path)
        else:
            wav_path = audio_path
        
        with open(wav_path, "rb") as audio_file:
            result = elevenlabs_client.speech_to_text.convert(
                audio=audio_file
            )
        
        return result.text
    except Exception as e:
        logger.error(f"ElevenLabs STT failed: {str(e)}")
        raise e

# Backup STT model: Whisper
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("medium") 
        logger.info("Whisper small model loaded")
    return whisper_model

# Can add more cleaning rules
def clean_transcribed_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # Need to add more cleaning rules
    return text

def whisper_stt(audio_path):
    model = get_whisper_model()
    result = model.transcribe(
        audio_path,
        language='en',
        task='transcribe',
        initial_prompt="This is a real estate conversation. The person is providing their name, phone number, and property address in Melbourne, Australia. Common street names include La Trobe Street, Wills Street, Spencer Street, Flinders Street, Elizabeth Street.",
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        word_timestamps=True
    )
    original_text = str(result['text']).strip()
    cleaned_text = clean_transcribed_text(original_text)
    logger.info(f"Original transcription (Whisper): {original_text}")
    logger.info(f"Cleaned transcription (Whisper): {cleaned_text}")
    return cleaned_text

def speech_to_text(audio_path):
    transcript = None
    # Step 1: Try ElevenLabs first
    try:
        transcript = elevenlabs_stt(audio_path, api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        logger.warning(f"ElevenLabs STT failed, fallback to Whisper: {e}")

    # Step 2: Clean and verify ElevenLabs results
    cleaned_text = clean_transcribed_text(transcript) if transcript else ""
    if cleaned_text and len(cleaned_text) >= 3:
        logger.info(f"Transcription (ElevenLabs): {cleaned_text}")
        return cleaned_text

    # If the ElevenLabs STT failed, use Whisper as backup
    else:
        result = whisper_stt(audio_path)
        return result


# ==================== Step 3: Text Extraction -> Azure OpenAI ====================
def extract_info(text: str) -> Dict[str, Any]:
    # Enhanced prompt to extract more detailed information
    prompt = (
        "Extract detailed client information from the real estate conversation below. "
        "Return ONLY a JSON object with the following structure:\n"
        "{\n"
        "  \"name\": \"client name\",\n"
        "  \"phone\": \"phone number\",\n"
        "  \"address\": \"specific property address if mentioned\",\n"
        "  \"preferences\": {\n"
        "    \"budget_range\": \"weekly rent budget (e.g., $400-600)\",\n"
        "    \"property_type\": \"apartment/house/studio\",\n"
        "    \"bedrooms\": \"number of bedrooms needed\",\n"
        "    \"location_preferences\": \"near transport/schools/hospitals/shops\",\n"
        "    \"must_have_features\": [\"parking\", \"balcony\", \"gym\", \"pool\"]\n"
        "  },\n"
        "  \"search_intent\": \"buying/renting/investing\"\n"
        "}\n"
        f"Text: {text}\n"
        "If any field is not mentioned, use null. Focus on extracting all relevant preferences."
    )
    
    response = openai_client.chat.completions.create(
        model="gpt4o_voicebot",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content  
    logger.info(f"OpenAI response content: {content}")
    
    # Check if content is empty
    if not content or content.strip() == "":
        logger.error("OpenAI returned empty content")
        return extract_basic_info(text)
    
    # Check if content looks like JSON
    if content.startswith('{') and content.endswith('}'):
        info = json.loads(content)
        logger.info(f"Enhanced client information: {info}")
        return info
    else:
        logger.error(f"Invalid JSON format: {content}")
        return extract_basic_info(text)

# Back up function when OpenAI fails
def extract_basic_info(text: str) -> Dict[str, Any]:
    """Extract basic information when OpenAI fails"""
    info = {
        "name": None,
        "phone": None,
        "address": None,
        "preferences": {},
    }
    
    # Extract name (multiple patterns)

    name_patterns = [
        r'my name is (\w+)',
        r"my name's (\w+)", 
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
        r"this is (\w+)",
        r"my name (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"This is (\w+)",
        r"This is calling from (\w+)"
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            info["name"] = name_match.group(1)
            break

    # Extract phone number - handle Australian mobile numbers (0400-0499)
    # Pattern for "0402 66286" or "0402-66286" or "040266286"
    phone_patterns = [
        r'(\d{4}[-\s]?\d{6})',  # 0402 662860 format
        r'(\d{4}[-\s]?\d{3}[-\s]?\d{3})',  # Standard 10-digit format
        r'(\d{4}[-\s]?\d{4}[-\s]?\d{2})',  # Alternative format
    ]
    
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            phone = phone_match.group(1).replace(' ', '').replace('-', '')
            # Validate it's a reasonable phone number length
            if len(phone) >= 8 and len(phone) <= 12:
                info["phone"] = phone
                break
    
    # Extract address
    address_match = re.search(r'(\d+\s+\w+\s+Street)', text, re.IGNORECASE)
    if address_match:
        info["address"] = address_match.group(1)
    
    logger.info(f"Basic info extracted: {info}")
    return info

# ==================== Step 4: Connect and search the property information in our SQL online database ====================
def query_property_azure(address: str) -> Any:
    """In the Azure SQL database, search the property information"""
    conn_str = (
        f"DRIVER={{{AZURE_SQL_CONFIG['driver']}}};"
        f"SERVER={AZURE_SQL_CONFIG['server']};"
        f"DATABASE={AZURE_SQL_CONFIG['database']};"
        f"UID={AZURE_SQL_CONFIG['username']};"
        f"PWD={AZURE_SQL_CONFIG['password']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    # Upgrading the matching strategy
    # Try multiple matching strategies
    search_terms = []
    
    # Strategy 1: Full address
    search_terms.append(address)
    
    # Strategy 2: Extract street name (e.g., "25 Wills Street" -> "Wills Street")
    street_match = re.search(r'\d+\s+(\w+\s+Street)', address, re.IGNORECASE)
    if street_match:
        search_terms.append(street_match.group(1))
    
    # Strategy 3: Extract just the street name without "Street" (e.g., "Wills")
    street_name_match = re.search(r'\d+\s+(\w+)\s+Street', address, re.IGNORECASE)
    if street_name_match:
        search_terms.append(street_name_match.group(1))
    
    # Try each search term
    for search_term in search_terms:
        cursor.execute("SELECT * FROM [House_Data] WHERE [name] LIKE ?", ('%' + search_term + '%',))
        result = cursor.fetchone()
        if result:
            logger.info(f"Found property with search term '{search_term}': {result[0]}")
            conn.close()
            return result
    
    conn.close()
    logger.info("Property not found")
    return "Not found"

# Return the information of the matched property
def property_summary(row: Any) -> str:
    """Generate the property information summary"""
    if row == "Not found":
        return "Sorry, no matching property found in our database. Please try a different address or contact AICG for assistance."
    
    return (
        f"Below is the best matched property for your search:\n"
        f"Address: {row[0]}\n"
        f"This property is a {row[1]}\n"
        f"It was built in {row[2]}, it is located in {row[18]}\n"
        f"It has{row[7]} bedrooms, {row[8]} bathrooms, {row[9]} car space(s)\n"
        f"Weekly rent: ${row[16]}\n"
        f"The property is located near {row[6]}, Postcode: {row[4]}.\n"
        f"For more details, please contact AICG team!"
    )

# ==================== Step 5: Text to speech -> ElevenLabs (Primary) / gTTS (Backup) ====================
# TTS 11labs 
def text_to_speech_elevenlabs(text: str, output_path: str = "result.mp3") -> str:
    """Text to speech using ElevenLabs (primary)"""
    try:
        # Generate audio with ElevenLabs
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Save the audio file
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        logger.info(f"Audio file generated (ElevenLabs): {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {str(e)}")
        # Fallback to gTTS
        return text_to_speech_gtts(text, output_path)

def text_to_speech_gtts(text: str, output_path: str = "result.mp3") -> str:
    """Text to speech using gTTS (backup)"""
    tts = gTTS(text, lang="en")
    tts.save(output_path)
    logger.info(f"Audio file generated (gTTS): {output_path}")
    return output_path

def text_to_speech(text: str, output_path: str = "result.mp3") -> str:
    """Main text to speech function - tries ElevenLabs first, falls back to gTTS"""
    return text_to_speech_elevenlabs(text, output_path)

# ==================== Step 6: Upload the client's information to Azure SQL ====================
def upload_to_azure_sql(data: Dict[str, Any]) -> bool:
    """Upload the client's information to the Azure SQL database"""
    conn_str = (
        f"DRIVER={{{AZURE_SQL_CONFIG['driver']}}};"
        f"SERVER={AZURE_SQL_CONFIG['server']};"
        f"DATABASE={AZURE_SQL_CONFIG['database']};"
        f"UID={AZURE_SQL_CONFIG['username']};"
        f"PWD={AZURE_SQL_CONFIG['password']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO Customer_Table (name, phone, address) VALUES (?, ?, ?)",
        data.get("name", ""), data.get("phone", ""), data.get("address", "")
    )
    conn.commit()
    conn.close()
    
    logger.info("Client information uploaded successfully")
    return True

# ==================== Main processing function ====================
@app.post("/process-audio") 
async def process_audio(file: UploadFile = File(...)):
    """Main processing function - integrate all 6 steps"""
    logger.info("Start processing the audio file...")
    
    # Step 1: Receive the front end audio
    audio_path = save_uploaded_audio(file)
    
    # Step 2: Audio to text
    text = speech_to_text(audio_path)
    
    # Step 3: Text information extraction
    client_info = extract_info(text)
    
    # Step 4: Database query
    property_info = ""
    if client_info.get('address'):
        result = query_property_azure(client_info['address'])
        property_info = property_summary(result)
    else:
        # when the client not provide an address, voice message reply
        property_info = (
            f"Hello {client_info.get('name', 'there')}! "
            "I'm your real estate assistant. Please provide a property address "
            "so I can help you find the perfect home. "
            "You can say something like 'I'm looking for a property at 63 La Trobe Street Melbourne'."
        )
    
    # Step 5: Text to speech
    # When the property information is not found, voice message reply
    if not property_info.strip():
        property_info = "I'm sorry, I couldn't understand your request. Please try again with a property address."
    audio_response_path = "result.mp3"
    text_to_speech(property_info, audio_response_path)
    
    # Step 6: Upload the client's information to Azure SQL
    if client_info:
        upload_to_azure_sql(client_info)
    
    # Clean the temporary files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    logger.info("Audio processing completed")
    
    return {
        "success": True,
        "recognized_text": text,
        "client_info": client_info,
        "property_info": property_info,
        "audio_file": "result.mp3",
        "audio_url": "/audio/result.mp3"
    }
    

# connection
# ==================== API interface ====================
# API THING!!!! 
# Return the audio file with the property information
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Get the generated audio file"""
    file_path = f"./{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "message": "Real Estate Voice Robot API is running"}

# Welcome audio
@app.get("/welcome-audio")
async def generate_welcome_audio():
    """Generate welcome audio message"""
    welcome_text = "Hi, I am your Real Estate AI assistant from AI consulting Group. If you provide your property preferences, I will help you recommend the most proper one for you?"
    
    # Generate audio file
    audio_path = "welcome.mp3"
    text_to_speech(welcome_text, audio_path)
    
    return {
        "success": True,
        "audio_file": "welcome.mp3",
        "audio_url": "/audio/welcome.mp3",
        "message": welcome_text
    }

# Get the available voices from ElevenLabs
@app.get("/elevenlabs/voices")
async def get_available_voices():
    """Get available ElevenLabs voices"""
    try:
        available_voices = elevenlabs_client.voices.get_all()
        
        # Format voice information
        voice_list = []
        for voice in available_voices.voices:
            voice_list.append({
                "voice_id": voice.voice_id,
                "name": voice.name,
                "category": getattr(voice, 'category', 'unknown'),
                "description": getattr(voice, 'description', ''),
                "labels": getattr(voice, 'labels', {})
            })
        
        return {
            "success": True,
            "voices": voice_list,
            "count": len(voice_list)
        }
    except Exception as e:
        logger.error(f"Failed to get voices: {str(e)}")
        # If the APi is not available, return the default voices
        default_voices = [
            {
                "voice_id": "ErXwobaYiN019PkySvjV",
                "name": "Antoni",
                "category": "cloned",
                "description": "Professional male voice",
                "labels": {"accent": "american", "gender": "male"}
            },
            {
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "name": "Rachel",
                "category": "cloned",
                "description": "Professional female voice",
                "labels": {"accent": "american", "gender": "female"}
            }
        ]
        return {
            "success": True,
            "voices": default_voices,
            "count": len(default_voices),
            "note": "Using default voices due to API permission limitation"
        }

class AudioGenerationRequest(BaseModel):
    text: str
    voice_id: str = ""
    model: str = "eleven_monolingual_v1"

@app.post("/elevenlabs/generate")
async def generate_custom_audio(request: AudioGenerationRequest = Body(...)):
    """Generate custom audio with specific voice and model"""
    try:
        # Use default voice if not specified
        voice_id = request.voice_id or ELEVENLABS_VOICE_ID
        
        # Generate audio
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=request.text,
            model_id=request.model
        )
        
        # Save to temporary file
        temp_path = f"custom_{hash(request.text) % 10000}.mp3"
        with open(temp_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        return {
            "success": True,
            "audio_file": temp_path,
            "audio_url": f"/audio/{temp_path}",
            "voice_id": voice_id,
            "model": request.model
        }
    except Exception as e:
        logger.error(f"Failed to generate custom audio: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "message": "Real Estate Voice Robot API", 
        "docs": "/docs",
        "endpoints": {
            "process_audio": "POST /process-audio - Process audio file",
            "get_audio": "GET /audio/{filename} - Get audio file",
            "health": "GET /health - Health check",
            "welcome_audio": "GET /welcome-audio - Generate welcome message",
            "elevenlabs_voices": "GET /elevenlabs/voices - Get available ElevenLabs voices",
            "elevenlabs_generate": "POST /elevenlabs/generate - Generate custom audio with ElevenLabs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Start the Real Estate Voice Robot Backend API...")
    print("API address: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Front end address: http://localhost:3000")
    print("=" * 50)
    # Backend
    uvicorn.run(app, host="0.0.0.0", port=8000)  