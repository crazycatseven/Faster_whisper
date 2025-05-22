from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import time
import tempfile
import shutil
from typing import Optional
from pydantic import BaseModel, Field

from faster_whisper import WhisperModel

# Model instance and info
default_model = None
default_model_info = {"size": "", "device": "", "loaded": False}

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API starting, loading default model...")
    load_model()
    yield
    print("API shutting down...")

app = FastAPI(
    title="Faster Whisper API", 
    description="Voice transcription API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_size="large-v3", device="cuda"):
    """Load or switch model"""
    global model, model_info
    
    try:
        if device == "cuda":
            print(f"Loading GPU model: {model_size}...")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            model_info = {"size": model_size, "device": "cuda", "loaded": True}
            return {"success": True, "message": f"Successfully loaded GPU model: {model_size}"}
        else:
            print(f"Loading CPU model: {model_size}...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            model_info = {"size": model_size, "device": "cpu", "loaded": True}
            return {"success": True, "message": f"Successfully loaded CPU model: {model_size}"}
    except Exception as e:
        if device == "cuda":
            try:
                print(f"GPU load failed: {e}")
                print("Fallback to CPU mode...")
                model = WhisperModel("small", device="cpu", compute_type="int8")
                model_info = {"size": "small", "device": "cpu", "loaded": True}
                return {"success": True, "message": f"GPU load failed, fallback to CPU model (small)"}
            except Exception as e2:
                model_info = {"size": "", "device": "", "loaded": False}
                return {"success": False, "error": f"Model load failed: {str(e2)}"}
        else:
            model_info = {"size": "", "device": "", "loaded": False}
            return {"success": False, "error": f"CPU model load failed: {str(e)}"}

class ModelRequest(BaseModel):
    model_size: str = Field(default="large-v3", description="Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)")
    device: str = Field(default="cuda", description="Device (cuda, cpu)")

@app.get("/")
def read_root():
    """API health check"""
    return {
        "status": "running",
        "model_loaded": model_info["loaded"],
        "model_info": model_info if model_info["loaded"] else None,
        "version": "1.0.0"
    }

@app.post("/load_model")
def api_load_model(request: ModelRequest):
    """Load or switch model"""
    result = load_model(request.model_size, request.device)
    return result

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    word_timestamps: bool = Form(True)
):
    """Transcribe uploaded audio file and return result"""
    global model, model_info
    
    # Ensure model is loaded
    if not model_info["loaded"]:
        load_result = load_model()
        if not load_result["success"]:
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1])
    try:
        shutil.copyfileobj(audio.file, temp_file)
        temp_file.close()
        lang_param = None if language == "auto" else language
        print(f"Transcribing file: {audio.filename}")
        start_time = time.time()
        segments_generator, info = model.transcribe(
            temp_file.name,
            language=lang_param,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
        transcribe_time = time.time() - start_time
        print(f"Transcribe time: {transcribe_time:.2f}s")
        print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
        segments_list = list(segments_generator)
        segments_data = []
        full_text = ""
        for segment in segments_list:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            if hasattr(segment, 'words') and segment.words:
                segment_dict["words"] = [
                    {"word": word.word, "start": word.start, "end": word.end}
                    for word in segment.words
                ]
            segments_data.append(segment_dict)
            full_text += segment.text + " "
        return {
            "success": True,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(transcribe_time, 2),
            "model": model_info["size"],
            "device": model_info["device"],
            "text": full_text.strip(),
            "segments": segments_data
        }
    except Exception as e:
        print(f"Transcribe error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove temp file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.get("/model_info")
def get_model_info():
    """Get current model info"""
    return model_info

# Main entry
if __name__ == "__main__":
    print("Starting Faster Whisper API...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=False,
    ) 