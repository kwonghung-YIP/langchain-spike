import logging
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama
from pydantic import BaseModel 

class ChatRequest(BaseModel):
    question: str

logger = logging.getLogger(__name__)
app = FastAPI()

model = ChatOllama(
    model="llama3.2",
    temperature="0.5",
)

@app.get("/echo/")
async def echo(name:str):
    return f"Hello {name}"

def dummy_stream():
    for i in range(20):
        yield f"data: {i}\n\n"
        time.sleep(1)

@app.get("/stream/")
async def simple_stream():
    return StreamingResponse(content=dummy_stream(),media_type="text/event-stream")

def model_stream_generator(messages):
    for chunk in model.stream(messages):
        logger.info(chunk)
        #yield chunk
        #content = chunk.replace("\n", "<br>")
        yield f"data: {chunk.content}\n\n"

@app.post("/chat/stream/")
def chat_stream(request:ChatRequest):
    messages = [
        ("user",request.question)
    ]
    return StreamingResponse(model_stream_generator(messages),media_type="text/event-stream")

async def model_astream_generator(messages):
    async for chunk in model.astream(messages):
        logger.info(chunk)
        #yield chunk
        #content = chunk.replace("\n", "<br>")
        yield f"data: {chunk.content}\n\n"

@app.post("/chat/astream/")
async def chat_stream(request:ChatRequest):
    messages = [
        ("user",request.question)
    ]
    return StreamingResponse(model_astream_generator(messages),media_type="text/event-stream")

@app.post("/chat/invoke/")
def chat_invoke(request:ChatRequest):
    response = model.invoke([
        ("user",request.question)
    ])
    return response

@app.post("/chat/ainvoke/")
async def chat_ainvoke(request:ChatRequest):
    response = await model.ainvoke([
        ("user",request.question)
    ])
    return response
