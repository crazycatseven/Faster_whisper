# Faster Whisper Audio Transcription API

An audio transcription tool based on [Faster Whisper](https://github.com/guillaumekln/faster-whisper), providing REST API interface for speech-to-text conversion.

## API Endpoints

- `GET /` - Health check
- `POST /transcribe` - Transcribe audio file
- `POST /load_model` - Load or switch model
- `GET /model_info` - Get current model information

## Example Usage

```python
import requests

# Transcribe audio file
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8080/transcribe",
        files={"audio": f},
        data={
            "language": "en",  # or "auto" for auto-detection
            "beam_size": 5,
            "vad_filter": True,
            "word_timestamps": True
        }
    )

result = response.json()
print(result["text"])
```
