How to check out and config the project
```bash
git clone https://github.com/kwonghung-YIP/langchain-spike
cd langchain-spike
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Update the requirements.txt after installed new packages
```bash
pip3 freeze > requirements.txt
```

Run the demo
```bash
# Agent loop with ollama
python3 01-agent-loop-ollama.py

# Agent loop with gemini2.0
export GOOGLE_API_KEY="your API key"
python3 02-agent-loop-gemini.py

# Agent loop with gpt-3.5-turbo
export OPENAI_API_KEY="your API key"
python3 02-agent-loop-gemini.py
```

Local vector store
```bash
# for redis
docker run -p 6379:6379 --name redis --rm redis/redis-stack-server:latest
# for pgvector
```

```bash
curl -v -s \
    http://localhost:8000/stream/

curl -v -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is prompt engineering? Could you give me a introduction with an example."}' \
    http://localhost:8000/chat/stream/

curl -v -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is prompt engineering? Could you give me a introduction with an example."}' \
    http://localhost:8000/chat/astream/

curl -v -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is prompt engineering? Could you give me a introduction with an example."}' \
    http://localhost:8000/chat/invoke/

curl -v -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is prompt engineering? Could you give me a introduction with an example."}' \
    http://localhost:8000/chat/ainvoke/
```