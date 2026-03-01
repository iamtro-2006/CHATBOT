import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.cloud import speech
from dotenv import load_dotenv
load_dotenv()
import os
print("GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
app = FastAPI(title="Realtime STT (Google Cloud Speech)", version="1.0.0")

SAMPLE_RATE = 16000
LANGUAGE_CODE_DEFAULT = "en-US"

@app.get("/")
def health():
    return {"status": "ok", "ws": "/ws/stt", "sample_rate": SAMPLE_RATE}

@app.websocket("/ws/stt")
async def ws_stt(websocket: WebSocket):
    """
    Client -> Server (binary frames): raw PCM16 little-endian audio @ 16kHz mono
    Client -> Server (text frames): JSON control, e.g. {"type":"start","lang":"vi-VN"} or {"type":"stop"}
    Server -> Client (text frames): JSON results: {"type":"result","is_final":bool,"transcript":str,"confidence":float|null}
    """
    await websocket.accept()

    speech_client = speech.SpeechClient()

    lang = LANGUAGE_CODE_DEFAULT
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
    stop_event = asyncio.Event()
    last_audio_ts = time.time()

    async def receiver():
        nonlocal lang
        try:
            while True:
                msg = await websocket.receive()

                if msg.get("bytes") is not None:
                    # audio chunk
                    chunk = msg["bytes"]
                    if chunk:
                        nonlocal last_audio_ts
                        last_audio_ts = time.time()
                        # avoid blocking forever if client spams
                        try:
                            audio_queue.put_nowait(chunk)
                        except asyncio.QueueFull:
                            # drop if overloaded
                            pass

                elif msg.get("text") is not None:
                    text = msg["text"]
                    try:
                        obj = json.loads(text)
                    except:
                        obj = {"type": text}

                    mtype = (obj.get("type") or "").lower()
                    if mtype == "start":
                        lang = obj.get("lang", lang) or lang
                    elif mtype == "stop":
                        break
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()

    def request_generator():
        # Stream audio only; config is passed separately to streaming_recognize
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                # keep stream alive with short silence if client is quiet too long
                if time.time() - last_audio_ts > 1.0:
                    silence = b"\x00\x00" * int(SAMPLE_RATE * 0.1)  # 100ms PCM16
                    yield speech.StreamingRecognizeRequest(audio_content=silence)
                time.sleep(0.02)
                continue
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    async def recognize_and_send():
        """
        streaming_recognize is blocking; run it in executor.
        """
        loop = asyncio.get_running_loop()

        def blocking_stream():
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=lang,
                enable_automatic_punctuation=True,
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False,
            )
            return speech_client.streaming_recognize(streaming_config, requests=request_generator())

        try:
            responses = await loop.run_in_executor(None, blocking_stream)
            for response in responses:
                for result in response.results:
                    alt = result.alternatives[0]
                    payload = {
                        "type": "result",
                        "is_final": bool(result.is_final),
                        "transcript": alt.transcript,
                        "confidence": getattr(alt, "confidence", None),
                    }
                    print(
                        "[STT]",
                        "FINAL" if result.is_final else "INTERIM",
                        alt.transcript,
                    )
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
        finally:
            stop_event.set()

    recv_task = asyncio.create_task(receiver())
    recog_task = asyncio.create_task(recognize_and_send())

    done, pending = await asyncio.wait({recv_task, recog_task}, return_when=asyncio.FIRST_COMPLETED)
    stop_event.set()
    for t in pending:
        t.cancel()

    try:
        await websocket.close()
    except:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
