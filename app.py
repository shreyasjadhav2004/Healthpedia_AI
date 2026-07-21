from flask import Flask, Response, render_template, request, stream_with_context
from src.helper import download_hugging_face_embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import json
import os
import glob
import requests
import subprocess
import time

from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Healthpedia AI is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


app = Flask(__name__)

load_dotenv()

embeddings = download_hugging_face_embeddings()

QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "medical-bot")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_data")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "qwen2.5:3b")
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def release_qdrant_lock():
    """Delete stale Qdrant .lock files before opening the database."""
    lock_files = glob.glob(os.path.join(QDRANT_PATH, "**", ".lock"), recursive=True)
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
            print(f"Removed stale lock: {lock_file}", flush=True)
        except OSError as e:
            print(f"Could not remove lock file {lock_file}: {e}", flush=True)


def stop_project_python_processes():
    if os.name != "nt":
        return 0

    current_pid = os.getpid()
    escaped_project_root = PROJECT_ROOT.replace("'", "''")
    command = f"""
$projectRoot = '{escaped_project_root}'
$currentPid = {current_pid}
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object {{
        $_.ProcessId -ne $currentPid -and
        $_.CommandLine -like "*$projectRoot*"
    }} |
    ForEach-Object {{
        Stop-Process -Id $_.ProcessId -Force
        $_.ProcessId
    }}
"""

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"Could not auto-close old Python processes: {exc}", flush=True)
        return 0

    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)

    stopped_pids = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]

    for pid in stopped_pids:
        print(f"Stopped old project Python process: {pid}", flush=True)

    return len(stopped_pids)


def get_qdrant_client(max_retries=2):
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return QdrantClient(path=QDRANT_PATH)
        except RuntimeError as exc:
            last_error = exc
            if "already accessed" not in str(exc) and "Storage folder" not in str(exc):
                raise

            print(f"\nQdrant local storage is busy: {QDRANT_PATH}", flush=True)
            if attempt >= max_retries:
                break

            print("Trying to close old Healthpedia Python processes ...", flush=True)
            stopped_count = stop_project_python_processes()
            if stopped_count == 0:
                print("No old Healthpedia Python processes were found.", flush=True)

            print("Waiting for the Qdrant lock to release ...", flush=True)
            time.sleep(2)

    raise SystemExit(
        f"\nQdrant local storage is still busy: {QDRANT_PATH}\n"
        "Close any running Healthpedia Python terminal and run this again.\n"
        "Local Qdrant cannot be opened by two Python processes at the same time.\n"
    ) from last_error


# ── Release stale lock BEFORE opening Qdrant ──────────────────────────────────
release_qdrant_lock()

# Loading the local Qdrant collection.
client = get_qdrant_client()
docsearch = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embeddings,   # ← "embedding" not "embeddings"
)

# Verify the collection is populated
try:
    collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
    print(f"✅ Collection '{QDRANT_COLLECTION_NAME}' loaded: {collection_info.points_count} vectors", flush=True)
except Exception as e:
    raise SystemExit(
        f"❌ Collection '{QDRANT_COLLECTION_NAME}' not found: {e}\n"
        "Run your ingestion script first!"
    )


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}


@app.route("/")
def index():
    return render_template('chat.html')


def stream_ollama_chat(prompt, model, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 256,
                "num_ctx": 2048,
            },
        },
        stream=True,
        timeout=(10, 300),
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content")
        if content:
            yield content
        if chunk.get("done"):
            break


def stream_ollama_generate(prompt, model, system_prompt=None):
    composed_prompt = prompt
    if system_prompt:
        composed_prompt = (
            f"System instructions:\n{system_prompt}\n\n"
            f"User prompt:\n{prompt}"
        )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": composed_prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 256,
                "num_ctx": 2048,
            },
        },
        stream=True,
        timeout=(10, 300),
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        chunk = json.loads(line)
        if chunk.get("response"):
            yield chunk["response"]
        if chunk.get("done"):
            break


def stream_ollama_response(prompt, system_prompt=None):
    errors = []

    for model in [OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL]:
        if not model:
            continue

        for stream_func in [stream_ollama_chat, stream_ollama_generate]:
            try:
                print(f"Using Ollama model: {model}", flush=True)
                yield from stream_func(prompt, model, system_prompt=system_prompt)
                return
            except requests.HTTPError as exc:
                response_text = exc.response.text if exc.response is not None else str(exc)
                errors.append(f"{model}: {response_text}")
                print(f"Ollama rejected {model}: {response_text}", flush=True)
            except requests.RequestException as exc:
                errors.append(f"{model}: {exc}")
                print(f"Ollama request failed for {model}: {exc}", flush=True)

    raise RuntimeError("No runnable Ollama model found. " + " | ".join(errors))


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)

    def generate_response():
        try:
            docs = docsearch.similarity_search(msg, k=2)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = PROMPT.format(context=context, question=msg)
            runtime_system_prompt = (
                f"{system_prompt}\n"
                "If the user asks outside health/medical scope, respond exactly with: "
                '"I can only help with health and medical topics. Please ask a medical question."'
            )

            for token in stream_ollama_response(prompt, system_prompt=runtime_system_prompt):
                print(token, end="", flush=True)
                yield token

            print("", flush=True)
        except Exception as exc:
            app.logger.exception("Error while generating response")
            yield (
                "\n\nSorry, I could not generate a response. "
                f"Server error: {type(exc).__name__}: {exc}"
            )

    return Response(
        stream_with_context(generate_response()),
        mimetype="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True)
