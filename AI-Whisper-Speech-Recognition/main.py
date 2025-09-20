import whisper
# Load the Whisper model
model = whisper.load_model("base")
# Transcribe the audio file
result = model.transcribe("C:/Users/HP/Website&App for portfolio/AI-Playground/AI-Whisper-Speech-Recognition/audios/aboutyou.mp3")
# Output the transcription
print(result["text"])