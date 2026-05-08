from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import subprocess
import time
from tqdm import tqdm

load_dotenv()

QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "medical-bot")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_data")
BATCH_SIZE = int(os.environ.get("QDRANT_BATCH_SIZE", "64"))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def stop_project_python_processes():
    if os.name != "nt":
        print(
            "Qdrant is busy. Auto-closing processes is only configured for Windows.",
            flush=True,
        )
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

            print("Trying to close old app.py/store_index.py processes ...", flush=True)
            stopped_count = stop_project_python_processes()
            if stopped_count == 0:
                print("No old project Python processes were found.", flush=True)

            print("Waiting for the Qdrant lock to release ...", flush=True)
            time.sleep(2)

    raise SystemExit(
        f"\nQdrant local storage is still busy: {QDRANT_PATH}\n"
        "Close any running app.py/store_index.py process and run this script again.\n"
        "Local Qdrant cannot be opened by two Python processes at the same time.\n"
    ) from last_error


def collection_exists(client, collection_name):
    collections = client.get_collections().collections
    return any(collection.name == collection_name for collection in collections)


print("Opening local Qdrant storage ...", flush=True)
client = get_qdrant_client()


print("Loading PDFs from data/ ...", flush=True)
extracted_data = load_pdf("data/")
print(f"Loaded {len(extracted_data)} PDF pages.", flush=True)

print("Splitting PDF text into chunks ...", flush=True)
text_chunks = text_split(extracted_data)
texts = [chunk.page_content for chunk in text_chunks]
metadatas = [chunk.metadata for chunk in text_chunks]
print(f"Created {len(text_chunks)} chunks.", flush=True)

print("Loading embedding model ...", flush=True)
embeddings = download_hugging_face_embeddings()
vector_size = len(embeddings.embed_query("healthpedia"))

if collection_exists(client, QDRANT_COLLECTION_NAME):
    print(f"Deleting old Qdrant collection: {QDRANT_COLLECTION_NAME}", flush=True)
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

print(f"Creating fresh Qdrant collection: {QDRANT_COLLECTION_NAME}", flush=True)
client.create_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    ),
)

docsearch = Qdrant(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embeddings=embeddings,
)

print("Embedding chunks and uploading to Qdrant ...", flush=True)
with tqdm(total=len(texts), desc="Indexing", unit="chunk") as progress:
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        docsearch.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            batch_size=BATCH_SIZE,
        )
        progress.update(len(texts[start:end]))

count = client.count(QDRANT_COLLECTION_NAME).count
print(f"Done. Indexed {count} chunks into '{QDRANT_COLLECTION_NAME}'.", flush=True)
