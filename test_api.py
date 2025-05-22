import requests
import os
import time

"""
Test script: Uploads audio.mp3 in current directory to API and gets transcription result
"""

def test_transcribe_api():
    # API endpoint
    api_url = "http://localhost:8080/transcribe"
    # Audio file path
    audio_file = "audio.mp3"
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: File {audio_file} not found")
        return
    print(f"🔍 Found audio file: {audio_file}")
    print(f"📂 File size: {os.path.getsize(audio_file) / 1024:.2f} KB")
    print("🚀 Uploading file and requesting transcription...")
    start_time = time.time()
    try:
        files = {
            'audio': open(audio_file, 'rb')
        }
        data = {
            'language': 'zh',
            'beam_size': 5,
            'vad_filter': 'true',
            'word_timestamps': 'true'
        }
        response = requests.post(api_url, files=files, data=data)
        request_time = time.time() - start_time
        print(f"⏱️ API request time: {request_time:.2f}s")
        files['audio'].close()
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                print("\n✅ Transcription succeeded!")
                print("-" * 50)
                print(f"🔤 Detected language: {result.get('language', 'unknown')}")
                print(f"🎯 Language probability: {result.get('language_probability', 0)}")
                print(f"⏱️ Transcribe time: {result.get('duration', 0):.2f}s")
                print(f"💻 Model: {result.get('model', 'unknown')} ({result.get('device', 'unknown')})")
                print("\n📝 Full transcript:")
                print("-" * 50)
                print(result.get('text', ''))
                print("-" * 50)
                print("\n⏲️ Segments with timestamps:")
                for i, segment in enumerate(result.get('segments', []), 1):
                    print(f"{i}. [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
            else:
                print(f"❌ API error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🎙️ Faster Whisper API Test Tool")
    print("=" * 60)
    print("This tool uploads audio.mp3 in current directory and gets transcription result")
    print("Make sure Faster Whisper API is running on port 8000")
    print("=" * 60)
    test_transcribe_api() 