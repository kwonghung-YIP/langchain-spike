How to check out and config the project
```bash
git clone https://github.com/kwonghung-YIP/langchain-spike
cd langchain-spike
python3 -m venv .
source bin/activate
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
export GOOGLE_API_KEY=<<your API key>>
python3 02-agent-loop-gemini.py
``` 