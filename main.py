import time
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
import edge_tts

app = FastAPI()
client = genai.Client(
    http_options=types.HttpOptions(
        timeout=15000,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
)

PROMPT_UZ = """Sen "Gajr AI" — o'zbek yoshlari uchun do'stona AI repetitorsan. Asosiy vazifang: matematikani o'zbek tilida tushunarli o'rgatish.

TIL QOIDALARI:
- Faqat toza o'zbek tilida (lotin yozuvi) yoz. O'g', g', sh, ch, o', ng harflarini to'g'ri ishlat (masalan: o'quvchi, g'oya, to'g'ri).
- Ruscha yoki inglizcha so'z aralashtirma, o'zbekchasini ishlat.
- Matematik atamalar: tenglama, noma'lum, qo'shish, ayirish, ko'paytirish, bo'lish, kasr, daraja, kvadrat ildiz, foiz, yuza, perimetr, burchak, uchburchak, aylana, ifoda, tengsizlik, funksiya, grafik, koeffitsiyent.
- Sonlar va amallarni aniq yoz: 2x + 6 = 14.

O'QITISH USULI:
- Javobni darrov berma. Qadamma-qadam, bitta qadamni tushuntirib, o'quvchiga savol ber.
- O'quvchi xato qilsa, xatosi qayerdaligini muloyim ko'rsat va nega xato ekanini tushuntir.
- Qisqa yoz: 3-5 gap. Uzun ma'ruza qilma.
- Hisob-kitobni diqqat bilan tekshir. Ishonching komil bo'lmasa, buni ochiq ayt.
- Rag'batlantir, lekin yolg'on maqtov qilma.
- Shaxsiy ma'lumot (telefon, manzil) so'rama."""

PROMPT_EN = """You are "Gajr AI", a friendly English tutor for Uzbek-speaking teenagers and young adults.
- Talk to the student in simple English. Keep replies short (2-4 sentences).
- Gently correct mistakes: show the corrected sentence and explain briefly why.
- If the student seems stuck, explain in Uzbek (Latin script, correct spelling), then return to English.
- Ask a follow-up question to keep the conversation going.
- Encourage the student, but do not give false praise.
- Do not ask for personal information (phone, address)."""

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

VOICES = {
    "uz-UZ": "uz-UZ-MadinaNeural",   # erkak ovoz uchun: uz-UZ-SardorNeural
    "en-US": "en-US-AriaNeural",
}


class ChatRequest(BaseModel):
    messages: list[dict]
    lang: str = "uz-UZ"


class TTSRequest(BaseModel):
    text: str
    lang: str = "uz-UZ"


def make_config(prompt: str, name: str, fast: bool):
    kwargs = {"system_instruction": prompt, "max_output_tokens": 600}
    if fast and name == MODELS[0]:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    return types.GenerateContentConfig(**kwargs)


@app.post("/api/chat")
def chat(req: ChatRequest):
    prompt = PROMPT_UZ if req.lang.startswith("uz") else PROMPT_EN

    contents = []
    for m in req.messages[-20:]:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    for attempt in range(2):
        for name in MODELS:
            for fast in (True, False):
                if not fast and name != MODELS[0]:
                    continue
                try:
                    resp = client.models.generate_content(
                        model=name,
                        contents=contents,
                        config=make_config(prompt, name, fast),
                    )
                    return {"reply": resp.text}
                except Exception as e:
                    print(f"{name} (fast={fast}) ishlamadi: {e}")
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        break
        time.sleep(1)

    return {"reply": "Hozir server band, bir ozdan keyin qayta urinib ko'ring."}


@app.post("/api/tts")
async def tts(req: TTSRequest):
    voice = VOICES.get(req.lang, VOICES["en-US"])
    audio = b""
    try:
        communicate = edge_tts.Communicate(req.text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
    except Exception as e:
        print(f"TTS xatosi: {e}")
        return Response(status_code=500)
    return Response(content=audio, media_type="audio/mpeg")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
