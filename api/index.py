from http.server import BaseHTTPRequestHandler
import json
import os
import base64
import tempfile
from pymongo import MongoClient
from groq import Groq

# ১. এনভায়রনমেন্ট ভেরিয়েবল লোড করা (Vercel থেকে)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# ২. Groq ক্লায়েন্ট ইনিশিয়ালাইজ
groq_client = Groq(api_key=GROQ_API_KEY)

# ৩. MongoDB ডেটাবেস কানেকশন
try:
    db_client = MongoClient(MONGO_URI)
    db = db_client["tx800_database"]
    collection = db["chat_history"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")

def transcribe_audio(base64_audio):
    """আপনার ভয়েসকে টেক্সটে রূপান্তর (Groq Whisper)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(base64.b64decode(base64_audio))
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_path, file.read()), model="whisper-large-v3", language="bn"
            )
        return transcription.text
    except Exception as e:
        print(f"Groq STT Error: {e}")
        return None

def get_ai_response(user_message, history):
    """Groq AI দিয়ে উত্তর তৈরি"""
    system_prompt = {
        "role": "system", 
        "content": "You are Tx 800 Mini, a shy, caring AI companion. Keep answers concise and friendly."
    }
    messages = [system_prompt] + history + [{"role": "user", "content": user_message}]
    
    try:
        # শুধুমাত্র Groq ব্যবহার করা হচ্ছে
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq AI Error: {e}")
        return "আমি এখন একটু ব্যস্ত আছি, পরে কথা বলি?"

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        user_id = data.get("user_id", "default_user")
        text_input = data.get("text", "")
        audio_base64 = data.get("audio", "")
        
        # যদি অডিও পাঠানো হয়, তবে টেক্সটে কনভার্ট করবে
        if audio_base64:
            user_message = transcribe_audio(audio_base64)
            if not user_message:
                user_message = "Sorry, I couldn't hear you."
        else:
            user_message = text_input

        if not user_message:
            self.send_response(400)
            self.end_headers()
            return

        # MongoDB থেকে আগের কথা রিকল করা (Memory)
        history = []
        try:
            past_chats = collection.find({"user_id": user_id}).sort("_id", -1).limit(5)
            for chat in reversed(list(past_chats)):
                history.append({"role": "user", "content": chat["user_message"]})
                history.append({"role": "assistant", "content": chat["bot_reply"]})
        except Exception as e:
            print(f"History Error: {e}")

        # AI থেকে উত্তর নেওয়া
        bot_reply = get_ai_response(user_message, history)

        # MongoDB তে নতুন চ্যাট সেভ করা
        try:
            collection.insert_one({
                "user_id": user_id,
                "user_message": user_message,
                "bot_reply": bot_reply
            })
        except Exception as e:
            print(f"Save Error: {e}")

        # রেসপন্স পাঠানো
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"reply": bot_reply, "user_text": user_message}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
