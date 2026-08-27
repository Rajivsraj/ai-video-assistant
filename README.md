I have used open source api keys, to run this project you would need those, so to get those go to images/output/img_key.png convert it in text, becuase git not allowing to push keys

# ai-video-assistant

![1.png](images/output/1.png)
![2.png](images/output/2.png)
![3.png](images/output/3.png)
![4.png](images/output/4.png)
![5.png](images/output/5.png)
![6.png](images/output/6.png)


[//]: # (Packages)
1. pydub
   - Download Dependencies
     - Run Command: winget install --id=Gyan.FFmpeg -e --source=winget
- t lets you load, slice, edit, and export audio files using an easy interface, with FFmpeg handling the heavy lifting. It’s widely used for tasks like trimming audio, adjusting volume, adding effects, and converting between formats.


Step 1
Download videos (from youtube or any other source, in .wav format)
        |
Change the khz and channel rate to 16k (to use Wishper AI)
        |
Chunk wav audio (10 min per chunk)


Step 2: AI Transcription (Will Run OpenAI Whisper locally to convert audio chunks into text
Install:
    - pip install whisper
WHAT THIS FILE DOES
- Loads the Whisper model once into memory (tiny/small/medium/large)
- Transcribes each audio chunk one by one
- Supports translate mode — converts Hindi speech directly to English text
- Combines all chunk transcripts into one single clean transcript
- Model size is configurable via .env — balances speed vs accuracy
Cons
- Whisper don't work for hindi translation properly we we can use SARVAM MODEL for that 


# FLOW
Download videos (from youtube or any other source, in .wav format)
        |
Change the khz and channel rate to 16k (to use Wishper AI)
        |
Chunk wav audio (10 min per chunk)
        |
AI Transcription (core/transcriber.py) translating audio to text
        |
Extractor
        |
Summerizer 

