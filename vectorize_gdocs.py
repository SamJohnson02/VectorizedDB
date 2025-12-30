# vectorize_gdocs.py

import os
import json
from typing import List, Dict

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from openai import OpenAI

from .env import GOOGLE_FOLDER_ID

# ---------------- CONFIG ----------------

# Google Drive scope: read-only
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Folder in Google Drive that contains the docs you want to embed.
# Get this from the URL: https://drive.google.com/drive/folders/<FOLDER_ID>

# OpenAI embedding model
EMBED_MODEL = "text-embedding-3-large"

# Output embeddings file
OUTPUT_JSONL = "gdocs_embeddings.jsonl"

# ----------------------------------------


def get_gdrive_service():
    """
    Authenticate and return a Google Drive API service object.
    Requires 'credentials.json' in the same directory on first run.
    """
    creds = None

    # Load existing token if available
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    return service


def list_docs_in_folder(service, folder_id: str) -> List[Dict]:
    """
    List Google Docs in the specified folder.
    """
    query = (
        f"'{folder_id}' in parents and "
        "mimeType='application/vnd.google-apps.document' "
        "and trashed=false"
    )

    docs: List[Dict] = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        docs.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)

        if not page_token:
            break

    return docs


def export_doc_as_text(service, file_id: str) -> str:
    """
    Export a Google Doc as plain text using the Drive API.
    """
    request = service.files().export(
        fileId=file_id,
        mimeType="text/plain"
    )
    data = request.execute()
    return data.decode("utf-8", errors="ignore")


def simple_chunk(text: str, max_chars: int = 1500) -> List[str]:
    """
    Naive chunking: break text into <= max_chars pieces,
    preferring to split on paragraph boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + "\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


def embed_text_chunks(client: OpenAI, chunks: List[str], metadata: Dict):
    """
    Call OpenAI embeddings on a list of text chunks.
    Yields JSON-serializable records ready to write to JSONL.
    """
    if not chunks:
        return

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks,
    )

    for i, emb in enumerate(response.data):
        yield {
            "doc_id": metadata["doc_id"],
            "doc_name": metadata["doc_name"],
            "chunk_index": i,
            "text": chunks[i],
            "embedding": emb.embedding,  # list[float]
        }


def main():
    # Init OpenAI client (uses OPENAI_API_KEY env var)
    client = OpenAI()

    # Init Google Drive service
    drive_service = get_gdrive_service()

    print("Listing docs in folder...")
    docs = list_docs_in_folder(drive_service, GOOGLE_FOLDER_ID)
    print(f"Found {len(docs)} docs.")

    if not docs:
        print("No docs found. Check GOOGLE_FOLDER_ID.")
        return

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for doc in docs:
            file_id = doc["id"]
            doc_name = doc["name"]
            print(f"Processing: {doc_name} ({file_id})")

            # Get text from Google Doc
            text = export_doc_as_text(drive_service, file_id)
            if not text.strip():
                print("  (empty doc, skipping)")
                continue

            # Chunk text
            chunks = simple_chunk(text, max_chars=1500)
            print(f"  Chunks: {len(chunks)}")

            # Embed chunks and write to JSONL
            for record in embed_text_chunks(
                client=client,
                chunks=chunks,
                metadata={"doc_id": file_id, "doc_name": doc_name},
            ):
                f_out.write(json.dumps(record) + "\n")

    print(f"Done! Embeddings written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()