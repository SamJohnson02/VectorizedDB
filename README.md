# Vectorized Google Docs Search

This project downloads Google Docs from a Google Drive folder, converts them to text, creates embeddings using OpenAI, and lets you run semantic (meaning-based) search over them.

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add Google credentials
Create a Google Cloud project → enable Google Drive API → create an OAuth Client (Desktop App) → download credentials.json → place it in this folder.

### 3. Add yourself as a test user
Google Cloud Console → OAuth consent screen → Test users → add your Gmail.

### 4. Get your Google Drive Folder ID
Open the folder in your browser:
https://drive.google.com/drive/folders/<FOLDER_ID>
Copy everything after /folders/ and paste it into vectorize_gdocs.py:
GOOGLE_FOLDER_ID = "your-folder-id"

### 5. Set your OpenAI API key
macOS/Linux:
export OPENAI_API_KEY="sk-..."
Windows PowerShell:
$env:OPENAI_API_KEY = "sk-..."

## Vectorize Your Docs
Run:
python3 vectorize_gdocs.py

This will:
- Authenticate with Google  
- Download all Google Docs in the folder  
- Convert them to text  
- Chunk them  
- Create embeddings  
- Save results to gdocs_embeddings.jsonl  

## Search Your Docs
Run:
python3 search_gdocs.py  
Enter a question and it will return the most semantically similar text chunks.

Example query:
PG&E interval ETL

## Files
- vectorize_gdocs.py — pulls Google Docs + creates embeddings  
- search_gdocs.py — semantic search  
- requirements.txt — dependencies  
- credentials.json — Google OAuth  
- token.json — auto-created  
- gdocs_embeddings.jsonl — your embedded documents  
