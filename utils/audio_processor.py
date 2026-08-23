import yt_dlp
from pydub import AudioSegment
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

# Ensure downloads folder is inside main project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # utils/
MAIN_DIR = os.path.dirname(BASE_DIR)                   # ai-video-assistant/
DOWNLOAD_DIR = os.path.join(MAIN_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Capture Audio from Video
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "cookiesfrombrowser": "chrome",  # new
        "jsexecutor": "node",            # new
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }
        ],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio file to wav format using pydub

    Converting downloaded .wav file again in .wav to change the kHz and mono.
    after that we will be able to use whisper AI
    """
    output_path = os.path.splitext(input_path)[0] + "_converted_16k_mono.wav"
    audio = AudioSegment.from_file(input_path)   # <-- FIXED: load input_path
    audio = audio.set_channels(1).set_frame_rate(16000)  # mono, 16kHz
    audio.export(output_path, format="wav")
    return output_path

# Example usage
# downloaded_file = download_youtube_audio("https://www.youtube.com/watch?v=jevuDDjFEsM&list=PLOspHqNVtKAC-FUNMq8qjYVw6_semZHw0&index=38")
# downloaded_file = download_youtube_audio("https://www.youtube.com/watch?v=UMYtqHptYvA")
# wav_file = convert_to_wav(downloaded_file)


# Chunk wav audio
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detect YouTube URL. Downloading audio")
        wav_path = download_youtube_audio(source)
    else:
        print("Detect Local file. Converting to wav")
        wav_path = convert_to_wav(source)

    print("Chunking audio .....")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created")

    return chunks

# x = chunk_audio(wav_file)
# print(x)

