# Quick start

This page starts the local PathRAG application. For configuration details, see [Installation](INSTALLATION.md).

## Backend

From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `sample.env` to `.env`, then set:

- `SECRET_KEY`: a unique random value for local authentication tokens
- `DEEPSEEK_API_KEY` or the compatible provider key used by your configuration
- `DEEPSEEK_API_BASE`, `DEEPSEEK_MODEL`, and embedding settings as needed

Start the API:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Check the API reference at [http://localhost:8000/docs](http://localhost:8000/docs).

## Web client

Open a second terminal:

```bash
cd ui
npm install
npm start
```

Visit [http://localhost:3000](http://localhost:3000), register an account, upload a document, and ask a question.

## Windows shortcuts

The repository includes `start-api.bat` and `start-ui.bat`. Review the scripts before running them; the API script installs dependencies and binds the service to all network interfaces.
