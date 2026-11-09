# Security

This repository is a research prototype and is not configured as a production service.

## Reporting a vulnerability

Do not post API keys, passwords, personal data, or exploitable details in a public issue. Use GitHub's private vulnerability reporting if it is enabled for this repository; otherwise contact the repository owner through the contact method on their GitHub profile.

## Safe local use

- Keep real credentials in an untracked `.env` file.
- Use a unique `SECRET_KEY`; never reuse the sample value.
- Restrict CORS origins and account registration before exposing the service.
- Treat uploaded documents, prompts, retrieved passages, logs, and local indexes as potentially sensitive.
- Confirm provider data policies before sending document text to hosted model APIs.
