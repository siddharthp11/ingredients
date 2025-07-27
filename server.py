from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import io
import os
from openai import OpenAI
from logging import getLogger

logger = getLogger(__name__)

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize OpenAI client
client = OpenAI(api_key=f"{os.getenv('OPENAI_API_KEY')}")


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    logger.info(f"Processing audio: {file.filename}")
    # Read uploaded file into memory
    contents = await file.read()

    # Convert audio to a format that OpenAI can process
    audio = AudioSegment.from_file(io.BytesIO(contents), format="webm")

    # Export to MP3 format (OpenAI prefers MP3)
    audio_buffer = io.BytesIO()
    audio.export(audio_buffer, format="mp3")
    audio_buffer.seek(0)

    try:
        # Create a temporary file with proper extension
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_buffer.getvalue())
            temp_file_path = temp_file.name

        # Transcribe the audio using OpenAI
        with open(temp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

        # Clean up temporary file
        os.unlink(temp_file_path)

        logger.info(f"Transcription: {transcript}")
        return {"transcription": transcript}

    except Exception as e:
        # Clean up temporary file in case of error
        if "temp_file_path" in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return {"error": f"Transcription failed: {str(e)}"}
