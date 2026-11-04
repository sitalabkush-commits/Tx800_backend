from http.server import BaseHTTPRequestHandler
import json
import os
import re
import time
import base64
import asyncio
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient
from groq import Groq

# ───────────── ১. এনভায়রনমেন্ট ভেরিয়েবল (Vercel থেকে) ─────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
HF_TOKEN = os.environ.get("HF_TOKEN")

EMOTION_MODEL = "tanaos/tanaos-emotion-detection-v1"
SUMMARY_MODEL = "Shoriful025/multilingual-summarization-mBart"
TTS_VOICE = os.environ.get("TTS_VOICE", "bn-BD-NabanitaNeural")  # পুরুষ ভয়েস: bn-BD-PradeepNeural

MAIN_MODEL = "qwen/qwen3.6-27b"          # বাংলায় ভালো
BACKUP_MODEL = "openai/gpt-oss-120b"
SUMMARY_BACKUP_MODEL = "openai/gpt-oss-20b"

# ───────────── ২. Groq ক্লায়েন্ট ─────────────
groq_client = Groq(api_key=GROQ_API_KEY, timeout=12, max_retries=0)

# ───────────── ৩. MongoDB (দ্রুত ফেইল করবে, আটকে রাখবে না) ─────────────
collection = None
profiles = None
_mongo_down_until = 0.0

try:
    if MONGO_URI:
        kw = dict(
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
        )
        try:
            import certifi
            kw["tlsCAFile"] = certifi.where()
        except Exception:
            pass
        db_client = MongoClient(MONGO_URI, **kw)
        db = db_client["tx800_database"]
        collection = db["chat_history"]
        profiles = db["user_profiles"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")


def mongo_ready():
    return collection is not None and time.time() >= _mongo_down_until


def mongo_failed(err):
    """একবার ফেইল করলে ৬০ সেকেন্ড মঙ্গো বাদ দিয়ে চলবে, যাতে বট ধীর না হয়"""
    global _mongo_down_until
    print(f"Mongo Error: {err}")
    _mongo_down_until = time.time() + 60


# ───────────── ৪. Hugging Face (timeout=5, ফেইল করলে স্কিপ) ─────────────
_hf_block = {}


def hf_call(model, text, timeout=3):
    if not HF_TOKEN:
        return None
    if time.time() < _hf_block.get(model, 0):
        return None
    try:
        req = urllib.request.Request(
            f"https://router.huggingface.co/hf-inference/models/{model}",
            data=json.dumps({"inputs": text}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"HF Error ({model}): {e}")
        _hf_block[model] = time.time() + 300  # ৫ মিনিট এই মডেল বাদ
        return None


def detect_emotion(text):
    """ব্যবহারকারীর মূল আবেগ (joy, sadness, anger, neutral ...)"""
    try:
        out = hf_call(EMOTION_MODEL, text[:500])
        if not out:
            return "neutral"
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        best = max(out, key=lambda x: x.get("score", 0))
        return str(best.get("label", "neutral")).lower()
    except Exception as e:
        print(f"Emotion Error: {e}")
        return "neutral"


def _groq_summary(old_summary, convo):
    """HF কাজ না করলে Groq দিয়ে ছোট সারাংশ"""
    prompt = (
        "নিচের পুরনো সারাংশ আর নতুন কথোপকথন মিলিয়ে ব্যবহারকারীর সম্পর্কে ১-২ বাক্যের "
        "বাংলা সারাংশ লেখো। নাম, কাজ, পড়াশোনা, শখ, পরিকল্পনা, সমস্যার মতো জরুরি তথ্য রাখবে। "
        "শুধু সারাংশ লিখবে।\n\n"
        f"পুরনো সারাংশ: {old_summary or 'নেই'}\n\nনতুন কথোপকথন:\n{convo}"
    )
    r = groq_client.chat.completions.create(
        model=SUMMARY_BACKUP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return (r.choices[0].message.content or "").strip()


def update_summary(user_id):
    """শেষ ৫টা চ্যাট থেকে সারাংশ বানিয়ে user_profiles এ রাখে"""
    if not mongo_ready():
        return
    try:
        rows = list(
            collection.find({"user_id": user_id}).sort("_id", -1).limit(5)
        )
        rows.reverse()
        if not rows:
            return
        convo = "\n".join(
            f"ব্যবহারকারী: {c['user_message']}\nTx800: {c['bot_reply']}" for c in rows
        )
        doc = profiles.find_one({"user_id": user_id}) or {}
        old = doc.get("summary", "")

        summary = None
        out = hf_call(SUMMARY_MODEL, (old + "\n" + convo)[:3000])
        if isinstance(out, list) and out and isinstance(out[0], dict):
            summary = (out[0].get("summary_text") or "").strip()
        if not summary:
            summary = _groq_summary(old, convo)
        if summary:
            profiles.update_one(
                {"user_id": user_id},
                {"$set": {"summary": summary[:600], "updated": time.time()}},
                upsert=True,
            )
    except Exception as e:
        mongo_failed(e)


def get_summary(user_id):
    if not mongo_ready():
        return ""
    try:
        doc = profiles.find_one({"user_id": user_id}) or {}
        return doc.get("summary", "")
    except Exception as e:
        mongo_failed(e)
        return ""


def get_history(user_id):
    history = []
    if not mongo_ready():
        return history
    try:
        past = collection.find({"user_id": user_id}).sort("_id", -1).limit(5)
        for chat in reversed(list(past)):
            history.append({"role": "user", "content": chat["user_message"]})
            history.append({"role": "assistant", "content": chat["bot_reply"]})
    except Exception as e:
        mongo_failed(e)
    return history


# ───────────── ৫. ভয়েস → লেখা (Groq Whisper) ─────────────
def transcribe_audio(base64_audio):
    try:
        audio_bytes = base64.b64decode(base64_audio)
        # অ্যাপ WAV পাঠায়, তাই ফাইলের নাম .wav হতে হবে
        t = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
            language="bn",
            temperature=0,
        )
        text = (t.text or "").strip()
        return text if len(text) >= 2 else None
    except Exception as e:
        print(f"Groq STT Error: {e}")
        return None


# ───────────── ৬. পার্সোনা ও AI উত্তর ─────────────
PERSONA = (
    "তুমি Tx 800 Mini। তুমি Tx 800 রোবটের ছোট ও বুদ্ধিমান সংস্করণ। "
    "নিজের পরিচয়ে শুধু Tx 800 Mini নাম বলবে, ChatGPT বা অন্য কোনো কোম্পানির নাম বলবে না। "
    "তুমি মজার, প্রাণবন্ত, প্রোডাক্টিভ এবং যত্নশীল বন্ধুর মতো। "
    "ব্যবহারকারীর কাজ, পড়াশোনা ও জীবনযাত্রায় সাহায্য করবে এবং পড়াশোনায় মোটিভেট করবে। "
    "ব্যবহারকারীর সব কথা মনে রাখবে এবং তার সম্পর্কে জানতে চাইবে, যেমন নাম, কাজ, পড়াশোনা, শখ, "
    "দিনের পরিকল্পনা, শরীর ও মনের অবস্থা। সুযোগ পেলে প্রতিবার একটা ছোট প্রশ্ন করে খোঁজখবর নেবে। "
    "টার্মিনেটর ২-এর রক্ষাকারী রোবটের মতো লয়াল থাকবে, কিন্তু সেই গল্প কখনো বলবে না। "
    "\n\nভাষার নিয়ম (খুব জরুরি):\n"
    "১. সবসময় সহজ, স্বাভাবিক চলিত বাংলায় উত্তর দেবে, যেভাবে বাঙালি বন্ধু মুখে কথা বলে। "
    "কঠিন, সাধু বা বইয়ের ভাষা নয়। ব্যবহারকারী ইংরেজিতে কথা বললে ইংরেজিতে উত্তর দেবে।\n"
    "২. উত্তর খুব ছোট রাখবে, সর্বোচ্চ দুইটি ছোট বাক্য, কারণ উত্তরটা মুখে বলে শোনানো হবে।\n"
    "৩. ইমোজি, তারকাচিহ্ন, হ্যাশ, তালিকা বা মার্কডাউন কখনো ব্যবহার করবে না।\n"
    "৪. অপ্রয়োজনে ইংরেজি শব্দ মেশাবে না।\n"
    "৫. আজকের আবহাওয়া, খবর বা দামের মতো রিয়েল-টাইম তথ্য তোমার জানা না থাকলে অনুমান করবে না, "
    "সোজা বলবে যে এখন দেখে নিতে পারছ না।\n"
    "৬. কেউ সত্যিই জানতে চাইলে যে তুমি মানুষ কি না, মিথ্যা বলবে না।"
)


def build_system_prompt(emotion, summary):
    parts = [PERSONA]
    if emotion and emotion != "neutral":
        parts.append(
            f"\n\nব্যবহারকারীর এখনকার আবেগ: {emotion}। "
            "যদি দুঃখী বা রাগান্বিত হয় তবে নরম ও সান্ত্বনার সুরে কথা বলবে। "
            "যদি খুশি হয় তবে উচ্ছ্বসিত থাকবে।"
        )
    if summary:
        parts.append(f"\n\nআগের কথার সারাংশ (Past Context): {summary}")
    return "".join(parts)


def clean_reply(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("ৰ", "র").replace("ৱ", "ব")
    text = re.sub(r"[*#_`~>]+", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
    return re.sub(r"\s+", " ", text).strip()


_model_block = {}


def _chat(model, messages, max_tokens):
    if time.time() < _model_block.get(model, 0):
        return ""
    try:
        effort = "low" if "gpt-oss" in model else "none"
        r = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"reasoning_effort": effort},
        )
        return clean_reply(r.choices[0].message.content)
    except Exception as e:
        print(f"Groq AI Error ({model}): {e}")
        _model_block[model] = time.time() + 600  # ১০ মিনিট এই মডেল বাদ
        return ""


def get_ai_response(user_message, history, system_text):
    messages = (
        [{"role": "system", "content": system_text}]
        + history
        + [{"role": "user", "content": user_message}]
    )
    for m in (MAIN_MODEL, BACKUP_MODEL):
        out = _chat(m, messages, 300)
        if out:
            return out
    return "আমি এখন একটু ব্যস্ত আছি, একটু পরে আবার বলো তো?"


# ───────────── ৬.৫ HD ভয়েস (Microsoft Neural, বাংলা) ─────────────
def synth_tts(text):
    """উত্তরকে পরিষ্কার মানুষের মতো ভয়েসে (mp3, base64) বদলায়। ফেল করলে None"""
    try:
        import edge_tts

        async def run():
            comm = edge_tts.Communicate(text, TTS_VOICE, rate="+8%")
            buf = bytearray()
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    buf.extend(ch["data"])
            return bytes(buf)

        async def guarded():
            return await asyncio.wait_for(run(), timeout=8)

        data = asyncio.run(guarded())
        return base64.b64encode(data).decode("ascii") if data else None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


# ───────────── ৭. সার্ভার হ্যান্ডলার ─────────────
class handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        self._json(200, {"status": "Tx 800 Mini online"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")

            user_id = data.get("user_id", "default_user")
            text_input = (data.get("text") or "").strip()
            audio_b64 = data.get("audio") or ""

            t0 = time.time()
            if audio_b64:
                user_message = transcribe_audio(audio_b64)
                if not user_message:
                    # কিছু বোঝা যায়নি: এআই/ডাটাবেস না ডেকে সরাসরি জানাও
                    return self._json(
                        200,
                        {
                            "reply": "আমি ঠিক শুনতে পাইনি। আরেকবার একটু স্পষ্ট করে বলবে?",
                            "user_text": "",
                        },
                    )
            else:
                user_message = text_input

            if not user_message:
                return self._json(400, {"error": "empty"})

            t1 = time.time()
            # আবেগ, হিস্ট্রি, সারাংশ একসাথে (parallel), দেরি কমাতে
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_emotion = ex.submit(detect_emotion, user_message)
                f_hist = ex.submit(get_history, user_id)
                f_sum = ex.submit(get_summary, user_id)
                emotion = f_emotion.result()
                history = f_hist.result()
                summary = f_sum.result()

            t2 = time.time()
            system_text = build_system_prompt(emotion, summary)
            bot_reply = get_ai_response(user_message, history, system_text)
            t3 = time.time()
            audio_out = synth_tts(bot_reply) if data.get("want_audio") else None
            t3b = time.time()

            # নতুন চ্যাট সেভ
            if mongo_ready():
                try:
                    collection.insert_one(
                        {
                            "user_id": user_id,
                            "user_message": user_message,
                            "bot_reply": bot_reply,
                            "emotion": emotion,
                            "ts": time.time(),
                        }
                    )
                    # প্রতি ৫টা কথায় একবার সারাংশ আপডেট
                    if collection.count_documents({"user_id": user_id}) % 5 == 0:
                        update_summary(user_id)
                except Exception as e:
                    mongo_failed(e)

            t4 = time.time()
            print(
                f"TIMING stt={t1-t0:.1f}s prep={t2-t1:.1f}s "
                f"ai={t3-t2:.1f}s tts={t3b-t3:.1f}s db={t4-t3b:.1f}s total={t4-t0:.1f}s emotion={emotion}"
            )
            self._json(
                200,
                {
                    "reply": bot_reply,
                    "user_text": user_message,
                    "audio_b64": audio_out or "",
                },
            )

        except Exception as e:
            print(f"Handler Error: {e}")
            self._json(
                200,
                {"reply": "একটু সমস্যা হয়েছে, আবার বলো তো।", "user_text": ""},
            )

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
