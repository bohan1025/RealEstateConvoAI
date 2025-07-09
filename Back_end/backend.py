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
from elevenlabs import ElevenLabs

# ElevenLabs configuration
ELEVENLABS_API_KEY = "sk_0e1aa08ba33f7aae3651ac26cd6ffb3fbaef8b9ae5decb05" 
ELEVENLABS_VOICE_ID = "ErXwobaYiN019PkySvjV"  # voice ID 

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

import warnings
import multiprocessing
import subprocess
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")


# Adding a conversation status variable
class ConversationState:
    def __init__(self):
        self.stage = "greeting"  # greeting -> basic_info -> preferences -> property_search -> details -> booking
        self.client_profile = {}
        self.conversation_history = []
        self.last_property_results = None
# Global state management (can be changed to Redis/database later)
conversation_sessions = {}  # session_id -> ConversationState


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

# ==================== Step 0: Initialize the conversation state ====================
def gpt_conversation_manager(user_text: str, session_id: str) -> dict:
    # Get or create conversation state
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationState()
    
    state = conversation_sessions[session_id]
    state.conversation_history.append({"role": "user", "content": user_text})
    
    # Build GPT prompt
    system_prompt = f"""You are a professional real estate AI assistant. Current conversation stage: {state.stage}

    Conversation flow:
    1. greeting - Welcome the user, understand basic needs
    2. basic_info - Collect name, phone, search intent
    3. preferences - Collect budget, property type, location preferences  
    4. confirmation - Confirm all collected information before searching
    5. property_search - Ask for specific property address and search
    6. property_details - Present property information and ask about viewing interest
    7. booking - Ask for booking a viewing appointment(Provide two choices: 1. Wednesday 10:00 AM, 2. Thursday 10:00 AM)
    8. complete - Conversation completed successfully

    Current client information: {json.dumps(state.client_profile, ensure_ascii=False)}

    CRITICAL: CONFIRMATION TO PROPERTY_SEARCH TRANSITION
    If current stage is "confirmation" and user confirms the information (says "Yes", "That's correct", "Yes, it is", etc.):
    1. Move to "property_search" stage
    2. Set "needs_property_search": true
    3. MUST extract address from client profile "location_preferences" and put it in "address" field
4. Response should be: "Great! I'll search for available apartments at [address] within your budget. Please hold on for a moment."

EXAMPLE for confirmation stage:
If client_profile has: "location_preferences": "442 Elizabeth Street"
Then extracted_info must include: "address": "Elizabeth Street"

PROPERTY SEARCH STAGE REQUIREMENTS:
- ALWAYS set "needs_property_search": true when transitioning from confirmation
- ALWAYS extract address from client profile if available
- If client said "442 Elizabeth Street", set "address": "442 Elizabeth Street"
- Backend will search database and return results automatically

STAGE TRANSITION RULES:
- greeting → basic_info: When user responds to greeting
- basic_info → preferences: When you have name and phone
- preferences → confirmation: When you have budget, location, and property type
- confirmation → property_search: When user confirms information (MUST set needs_property_search: true)
- property_search → property_details: When property information is found and presented
- property_details → booking: When user expresses interest in viewing
- booking → complete: When viewing appointment is confirmed

CRITICAL: PROPERTY_DETAILS TO BOOKING TRANSITION
If current stage is "property_details" and user says any of these:
- "That would be great"
- "Do you have any choices for inspection?"
- "Yes, I'm interested"
- "I'd like to schedule a viewing"
- "When can I see it?"
- Any positive response about viewing
Then: Move to "booking" stage and offer viewing times.

BOOKING STAGE REQUIREMENTS:
- Offer exactly two choices: "Wednesday 10:00 AM" and "Thursday 10:00 AM"
- Ask user to choose one
- Once confirmed, move to complete stage

Please analyze the user's input and return JSON format:
{{
  "extracted_info": {{
    "name": "customer name or null",
    "phone": "phone number or null", 
    "budget_range": "weekly rent budget or null",
    "property_type": "apartment/house/studio or null",
    "bedrooms": "number of bedrooms or null",
    "location_preferences": "preferred locations or null",
    "search_intent": "buying/renting/investing or null",
    "must_have_features": ["feature1", "feature2"] or null,
    "viewing_time": "confirmed appointment time or null",
    "address": "specific property address or null"
  }},
  "next_stage": "next stage",
  "action": "continue_conversation|search_property|complete_conversation",
  "response": "response to the user",
  "needs_property_search": true/false,
  "conversation_complete": true/false
}}

IMPORTANT: When user confirms in confirmation stage, you MUST:
1. Set "next_stage": "property_search"
2. Set "needs_property_search": true
3. Extract address from client profile and put in "address" field
4. This will trigger database search and return results automatically

CRITICAL: In property_details stage, clearly present property information and ask about viewing interest.
"""

    # Call GPT
    response = openai_client.chat.completions.create(
        model="gpt4o_voicebot",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation history: {state.conversation_history[-3:]}\\n\\nUser's latest input: {user_text}"}
        ],
        temperature=0.2
    )
    
    # Parse GPT response
    try:
        gpt_analysis = json.loads(response.choices[0].message.content)
        
        # Update client information
        extracted_info = gpt_analysis.get("extracted_info", {})
        state.client_profile.update({k: v for k, v in extracted_info.items() if v})
        
        # Update conversation stage
        state.stage = gpt_analysis.get("next_stage", state.stage)
        
        # Add AI response to history
        ai_response = gpt_analysis.get("response", "")
        state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return gpt_analysis
        
    except json.JSONDecodeError:
        return {
            "action": "continue_conversation",
            "response": "Please tell me your needs, I will help you find a suitable property.",
            "needs_property_search": False,
            "conversation_complete": False
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


# ==================== Step 3: Connect and search the property information in our SQL online database ====================
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

# ==================== Step 4: Text to speech -> ElevenLabs (Primary) / gTTS (Backup) ====================
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

# ==================== Step 5: Upload the client's information to Azure SQL ====================
def upload_to_azure_sql(data: Dict[str, Any], recommended_property: str = None) -> bool:
    """Upload complete client information to Azure SQL database"""
    
    # Database connection
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
    
    # Enhanced data extraction with proper field mapping
    customer_name = data.get("name", "not given")
    phone_number = data.get("phone", "not given")
    budget_range = data.get("budget_range", "not given")
    property_type = data.get("property_type", "not given") 
    location_preferences = data.get("location_preferences", "not given")
    search_intent = data.get("search_intent", "not given")
    recommended_property_address = recommended_property if recommended_property else "not given"
    viewing_appointment_time = data.get("viewing_time", "not given")
    
    # Handle must_have_features list
    must_have_features = data.get("must_have_features", [])
    if isinstance(must_have_features, list) and must_have_features:
        must_have_features = ", ".join(must_have_features)
    else:
        must_have_features = "not given"
    
    # Handle bedrooms
    bedrooms = data.get("bedrooms", "not given")
    if bedrooms:
        bedrooms_str = f"{bedrooms} bedrooms"
    else:
        bedrooms_str = "not given"
    
    # Combine location preferences and bedrooms if available
    if bedrooms_str != "not given" and location_preferences != "not given":
        location_preferences = f"{location_preferences}, {bedrooms_str}"
    elif bedrooms_str != "not given":
        location_preferences = bedrooms_str
    
    conversation_stage = data.get("conversation_stage", "not given")
    conversation_summary = "Multi-round conversation completed"
    ai_interaction_status = "completed"
    
    # Check if customer already exists
    cursor.execute("SELECT id FROM All_Clients WHERE phone_number = ?", (phone_number,))
    existing_customer = cursor.fetchone()
    
    if existing_customer:
        # Update existing customer
        update_query = """
        UPDATE All_Clients SET 
            customer_name = ?,
            budget_range = ?,
            property_type = ?,
            location_preferences = ?,
            search_intent = ?,
            recommended_property_address = ?,
            viewing_appointment_time = ?,
            must_have_features = ?,
            conversation_stage = ?,
            conversation_summary = ?,
            ai_interaction_status = ?,
            updated_date = GETDATE()
        WHERE phone_number = ?
        """
        cursor.execute(update_query, (
            customer_name,
            budget_range,
            property_type,
            location_preferences,
            search_intent,
            recommended_property_address,
            viewing_appointment_time,
            must_have_features,
            conversation_stage,
            conversation_summary,
            ai_interaction_status,
            phone_number
        ))
        logger.info(f"Updated existing customer: {customer_name}")
    else:
        # Insert new customer
        insert_query = """
        INSERT INTO All_Clients (
            customer_name,
            phone_number,
            budget_range,
            property_type,
            location_preferences,
            search_intent,
            recommended_property_address,
            viewing_appointment_time,
            must_have_features,
            conversation_stage,
            conversation_summary,
            ai_interaction_status,
            created_date,
            updated_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
        """
        cursor.execute(insert_query, (
            customer_name,
            phone_number,
            budget_range,
            property_type,
            location_preferences,
            search_intent,
            recommended_property_address,
            viewing_appointment_time,
            must_have_features,
            conversation_stage,
            conversation_summary,
            ai_interaction_status
        ))
        logger.info(f"Inserted new customer: {customer_name}")
    
    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    
    return True

# Return the matched property information to the client  
def integrate_property_info_with_gpt(property_summary: str, state: ConversationState) -> str:
    prompt = f"""Based on the property information, generate a detailed and engaging response to the customer.

Client information: {json.dumps(state.client_profile, ensure_ascii=False)}
Conversation stage: {state.stage}
Property information: {property_summary}

Requirements:
1. Present the property details in a clear, engaging manner
2. Include key information: address, price, bedrooms, bathrooms, features
3. Highlight features that match customer preferences (budget, location, property type)
4. Use enthusiastic but professional tone
5. End with: "Would you like to schedule a viewing for this property?"
6. Make it conversational and natural, but not too long

Example format:
"I found a great property for you! It's a [bedrooms] bedroom [property_type] located at [address]. The weekly rent is [price], which fits your budget of [budget_range]. 

Key features include: [list main features]

This property is in [location] which matches your preference for [location_preferences]. 

Would you like to schedule a viewing for this property?"

Please return the reply content directly to the customer, do not use JSON format."""
# Use gpt to get the response
    response = openai_client.chat.completions.create(
        model="gpt4o_voicebot",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content 

# ==================== Main processing function ====================
@app.post("/process-audio") 
async def process_audio(file: UploadFile = File(...)):
    """GPT-driven multi-turn conversation processing"""
    
    # 1. Audio to text
    audio_path = save_uploaded_audio(file)
    text = speech_to_text(audio_path)
    
    # 2. Get session_id (simplified version)
    session_id = "default_session"
    
    # 3. GPT analysis and decision
    gpt_decision = gpt_conversation_manager(text, session_id)
    
    # 4. Initialize variables
    recommended_property_address = None
    response_text = ""
    
# 5. Check if we need to search for property
    if gpt_decision.get("needs_property_search") and gpt_decision.get("extracted_info", {}).get("address"):
    # Query property information
        address = gpt_decision["extracted_info"]["address"]
        property_result = query_property_azure(address)
    
    # Update conversation state
        state = conversation_sessions[session_id]
        state.last_property_results = property_result
    
    # Store recommended property address if found
        if property_result != "Not found":
            recommended_property_address = property_result[0]  # First column is address
        
        # Let GPT generate a reply based on the query result
            property_summary_text = property_summary(property_result)
        
        # Call GPT again to integrate property information
            final_response = integrate_property_info_with_gpt(property_summary_text, state)
            response_text = final_response
        
        # Automatically transition to property_details stage
            state.stage = "property_details"
        else:
            response_text = "Sorry, I couldn't find information about that property. Could you please provide another address?"
    else:
        # Use GPT's direct response
        response_text = gpt_decision.get("response", "")

    
    # 6. Check if conversation is complete
    conversation_complete = gpt_decision.get("conversation_complete", False)
    if conversation_complete:
        # Add a final goodbye message
        response_text += "\n\nThank you for using our service! Our team will contact you soon regarding your viewing appointment. Have a great day!"
        
        # Mark session for cleanup
        state = conversation_sessions[session_id]
        state.stage = "complete"
    
    # 7. Text to speech
    text_to_speech(response_text, "result.mp3")
    
    # 8. Upload complete client information to database
    state = conversation_sessions[session_id]
    
    # Prepare complete client data
    complete_client_data = state.client_profile.copy()
    complete_client_data["conversation_stage"] = state.stage
    
    # Check if we have essential information before uploading
    has_name = complete_client_data.get("name")
    has_phone = complete_client_data.get("phone")
    
    if has_name or has_phone:
        upload_success = upload_to_azure_sql(complete_client_data, recommended_property_address)
        if upload_success:
            logger.info("Complete client information uploaded to database")
        else:
            logger.error("Failed to upload client information to database")
    else:
        logger.info("No essential client information to upload yet")
    
    # 9. Clean up temporary files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    # 10. Return the result
    return {
        "success": True,
        "recognized_text": text,
        "ai_response": response_text,
        "conversation_stage": conversation_sessions[session_id].stage,
        "client_profile": conversation_sessions[session_id].client_profile,
        "audio_url": "/audio/result.mp3",
        "conversation_complete": conversation_complete,
        "recommended_property": recommended_property_address
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
    welcome_text = "Hi, I am your Real Estate AI assistant from AI consulting Group. I am willing to help you to find the best property. Can I get your name please?"
    
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