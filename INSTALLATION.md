# Installation and configuration

PathRAG is a local research prototype with a Python API and React web client. The commands below are intended for development on a trusted machine.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- A compatible chat-completion and embedding service

The Python dependencies include machine-learning packages and can require substantial download space.

## 1. Create the backend environment

From the repository root:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
& .venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure environment variables

Copy `sample.env` to `.env`. Generate a new secret rather than reusing the example:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set the printed value as `SECRET_KEY`. Configure the model and embedding endpoint, model names, and API key for your provider. The checked-in sample uses placeholder values only.

Common settings:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Signs authentication tokens; use a private, randomly generated value |
| `DEEPSEEK_API_KEY` | Example API credential consumed by the default model configuration |
| `DEEPSEEK_API_BASE` | OpenAI-compatible chat API base URL |
| `DEEPSEEK_MODEL` | Chat model name |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_API_BASE` | Embedding endpoint |
| `DATABASE_URL` | Relational metadata database; defaults to local SQLite |
| `WORKING_DIR` | PathRAG indexes and working files |
| `UPLOAD_DIR` | Uploaded documents |
| `CORS_ORIGINS` | Comma-separated browser origins; local default is `http://localhost:3000` |

Never commit `.env`, API keys, uploaded files, databases, or generated indexes.

## 3. Start the API

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the OpenAPI interface.

## 4. Start the web client

In a separate terminal:

```bash
cd ui
npm install
npm start
```

The development client is served at [http://localhost:3000](http://localhost:3000). Its default API address is `http://localhost:8000`; override it with `REACT_APP_API_URL` when needed.

## 5. First use

1. Register an account from the web interface.
2. Upload a document you are authorized to process.
3. Wait for ingestion to finish.
4. Ask a question and inspect the answer and available evidence views.

The repository intentionally does not create accounts with shared default passwords.

## Runtime data

The database, uploaded documents, logs, and graph/vector indexes are created locally according to `.env`. Back up or remove these directories explicitly when needed. Hosted model providers may receive prompts and retrieved document text. Check the provider's data handling terms before using sensitive material.

## Deployment note

Do not expose the development configuration directly to the public internet. Before deployment, configure HTTPS, a restricted CORS allowlist, secure secrets, controlled registration, rate limits, persistent storage, backups, and monitoring.
