# 🤗 Hugging Face Model Downloader

A lightweight Qt 6 GUI for searching Hugging Face repositories and downloading individual files or entire model repositories with progress tracking and resumable downloads.

## Features
- Search models by name (fuzzy) or enter a full `org/repo` identifier.  
- Browse repository file list with human‑readable sizes.  
- Download selected files **or** the whole repository.  
- Resumable downloads and size verification.  
- Parallel HEAD requests for fast size discovery.  
- Native look on macOS, Windows, and Linux.

## Installation

```bash
# Clone the repository
git clone https://github.com/pramjana/HF-Downloader.git
cd HF_Downloader

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Dependencies** (minimum):
> ```
> PySide6
> huggingface_hub
> requests
> ```

## Usage

```bash
python hf_downloader.py
```

1. Enter a model name (e.g., `bert-base-uncased`) or a full repo ID (`AIDC-AI/Ovis2.5-2B`).  
2. If the query is fuzzy, pick the desired repository from the drop‑down list.  
3. Select one or more files in the table, or click **Download Entire Repo**.  
4. Choose a destination folder.  
5. Watch the status bar and progress bars for download progress.

## Requirements
- Python 3.8 or newer  
- Internet connection (for Hub API calls and file downloads)  
- Qt 6 runtime (provided by `PySide6`)

## License
This project is licensed under the **Apache License 2.0**.
