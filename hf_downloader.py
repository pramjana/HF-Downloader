import sys
import threading
import shutil
import time
import requests
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PySide6.QtCore import Qt, Signal, QObject, QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QComboBox,
    QHeaderView,
    QGroupBox,
)

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    snapshot_download,
    list_repo_files,
    ModelInfo,
)

# ----------------------------------------------------------------------
# Helper: obtain size of a remote file via a HEAD request (fast, no body)
# ----------------------------------------------------------------------
def fetch_remote_size(url: str) -> int:
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        return int(resp.headers.get("content-length", 0))
    except Exception:
        return 0

# ----------------------------------------------------------------------
# Helper: stream a file from the Hub while emitting progress signals
# ----------------------------------------------------------------------
def stream_download(url: str, dest_path: Path, signals: "WorkerSignals") -> int:
    """Download *url* to *dest_path* in chunks and emit progress signals."""
    # First check if file exists and sizes match
    if dest_path.exists():
        local_size = dest_path.stat().st_size
        remote_size = fetch_remote_size(url)
        
        if remote_size > 0 and local_size == remote_size:
            signals.progress.emit(f"Skipped {dest_path.name} (already exists and size matches)")
            signals.progress_percent.emit(100)
            return remote_size
        else:
            signals.progress.emit(f"File exists but size mismatch: local {local_size} vs remote {remote_size}. Downloading...")

    # Check if we're resuming a partial download
    resume_download = False
    initial_size = 0
    if dest_path.exists():
        initial_size = dest_path.stat().st_size
        if initial_size > 0:
            resume_download = True

    # Proceed with download if file doesn't exist or sizes don't match
    headers = {}
    if resume_download:
        headers = {"Range": f"bytes={initial_size}-"}
        
    with requests.get(url, stream=True, timeout=30, headers=headers) as r:
        r.raise_for_status()
        
        # When resuming, total size is initial_size + remaining content
        if resume_download:
            if r.status_code == 206:  # Partial Content
                total = initial_size + int(r.headers.get("content-length", 0))
            else:
                # Server doesn't support resuming, start over
                initial_size = 0
                total = int(r.headers.get("content-length", 0))
        else:
            total = int(r.headers.get("content-length", 0))

        downloaded = initial_size
        start = time.time()

        # Ensure parent directory exists before writing
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file in append mode if resuming, write mode otherwise
        mode = "ab" if resume_download else "wb"
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                signals.progress.emit(
                    f"{downloaded / (1024**2):.2f} MiB / {total / (1024**2):.2f} MiB"
                )
                if total:
                    signals.progress_percent.emit(int(downloaded * 100 / total))

        elapsed = time.time() - start
        speed = (downloaded - initial_size) / (1024**2) / elapsed if elapsed > 0 else 0
        signals.progress.emit(f"Done ({speed:.2f} MiB/s, {elapsed:.1f}s)")
        signals.progress_percent.emit(100)

        return total



# ----------------------------------------------------------------------
# Helper: parallel HEAD requests (max 8 workers) – returns dict {url: size}
# ----------------------------------------------------------------------
def fetch_sizes_parallel(urls: List[str], max_workers: int = 8) -> Dict[str, int]:
    """Concurrent HEAD requests; returns mapping url → size (bytes)."""
    sizes: Dict[str, int] = {}

    def _head(u: str) -> (str, int):
        return u, fetch_remote_size(u)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_head, u): u for u in urls}
        for future in as_completed(future_to_url):
            u, sz = future.result()
            sizes[u] = sz
    return sizes


# ----------------------------------------------------------------------
# Signals used for communication between worker threads and the GUI
# ----------------------------------------------------------------------
class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)                # per‑file status line
    progress_percent = Signal(int)       # per‑file 0‑100 bar
    overall_progress = Signal(str)       # overall repo status line
    overall_progress_percent = Signal(int)  # overall repo 0‑100 bar
    combo_ready = Signal(list)           # list[ModelInfo] for search results
    files_ready = Signal(list)           # list[dict] – each dict has name & size


# ----------------------------------------------------------------------
# Main window – native look & feel (macOS respects system dark/light mode)
# ----------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤗 Hugging Face Model Downloader")
        self.setMinimumSize(720, 560)

        # --------------------------------------------------------------
        # 1️⃣  UI widgets (must exist before signals are connected)
        # --------------------------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setVisible(False)

        self.overall_label = QLabel("")
        self.status_bar = QLabel("")
        self.repo_label = QLabel("Repository: —")

        # Search area – inside a group box for visual separation
        self.search_group = QGroupBox("Search Model / Repository")
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "e.g. bert-base-uncased or AIDC-AI/Ovis2.5-2B"
        )
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_repo)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)
        self.search_group.setLayout(search_layout)

        self.match_combo = QComboBox()
        self.match_combo.setVisible(False)
        self.match_combo.currentIndexChanged.connect(self._combo_selected)

        # Table – filename column stretches, size column centered and auto‑sized
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["File", "Size"])
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)          # filename
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # size
        self.file_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.ExtendedSelection)

        # Buttons
        self.download_file_btn = QPushButton("Download Selected Files")
        self.download_file_btn.setEnabled(False)
        self.download_file_btn.clicked.connect(self.download_selected_files)

        self.download_repo_btn = QPushButton("Download Entire Repo")
        self.download_repo_btn.setEnabled(False)
        self.download_repo_btn.clicked.connect(self.download_entire_repo)

        # --------------------------------------------------------------
        # 2️⃣  Hub client & custom signal object
        # --------------------------------------------------------------
        self.api = HfApi()
        self.signals = WorkerSignals()
        self.signals.error.connect(self.show_error)
        self.signals.progress.connect(self.update_status)

        # --------------------------------------------------------------
        # 3️⃣  Connect signals that need the widgets created above
        # --------------------------------------------------------------
        self.signals.progress_percent.connect(self.progress_bar.setValue)
        self.signals.overall_progress.connect(self.update_overall_status)
        self.signals.overall_progress_percent.connect(
            self.overall_progress_bar.setValue
        )

        # Additional signals for search results and file listings
        self.signals.combo_ready.connect(self._on_combo_ready)
        self.signals.files_ready.connect(self._on_files_ready)

        # --------------------------------------------------------------
        # 4️⃣  Layout construction – clean, spaced, and platform‑native
        # --------------------------------------------------------------
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.search_group)
        main_layout.addWidget(self.match_combo)
        main_layout.addWidget(self.repo_label)

        main_layout.addWidget(QLabel("Files in repository (size shown):"))
        main_layout.addWidget(self.file_table)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.download_file_btn)
        btn_layout.addWidget(self.download_repo_btn)
        main_layout.addLayout(btn_layout)

        # Status & progress
        main_layout.addWidget(self.status_bar)
        main_layout.addWidget(self.progress_bar)

        # Overall repo progress (shown only during full‑repo download)
        main_layout.addWidget(self.overall_label)
        main_layout.addWidget(self.overall_progress_bar)

        self.setLayout(main_layout)

        # ------------------------------------------------------------------
        # Internal helpers for repo‑wide size tracking & simple cache
        # ------------------------------------------------------------------
        self.repo_total_bytes = 0
        self.repo_downloaded_bytes = 0
        self._size_cache: Dict[str, int] = {}   # url → size (kept for the session)

        # Apply a slightly larger, platform‑native font (no forced palette)
        self.apply_native_font()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def apply_native_font(self):
        """Use the system default font but increase point size a little for readability."""
        font = QFont()
        font.setPointSize(11)   # works well on macOS, Windows, Linux
        self.setFont(font)

    def update_status(self, msg: str):
        self.status_bar.setText(msg)

    def update_overall_status(self, msg: str):
        self.overall_label.setText(msg)

    def show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------
    # Search logic – exact repo, fuzzy name, and combo selection
    # ------------------------------------------------------------------
    def search_repo(self):
        query = self.search_input.text().strip()
        if not query:
            self.show_error("Please enter a model name or full repo id.")
            return

        # Reset UI
        self.repo_label.setText("Repository: searching …")
        self.file_table.setRowCount(0)
        self.download_file_btn.setEnabled(False)
        self.download_repo_btn.setEnabled(False)
        self.match_combo.clear()
        self.match_combo.setVisible(False)
        self.overall_label.clear()
        self.overall_progress_bar.setVisible(False)

        # Full repo supplied (contains a slash) → skip fuzzy search
        if "/" in query:
            self._finalize_repo_selection(query)
            return

        def _search():
            try:
                # Use API search parameter instead of local filtering
                models = list(self.api.list_models(search=query, limit=50))
                
                if not models:
                    # Fallback to REST search if API returns nothing
                    url = "https://huggingface.co/api/models"
                    params = {"search": query, "limit": 50}
                    resp = requests.get(url, params=params, timeout=10)
                    resp.raise_for_status()
                    items = resp.json()
                    models = [
                        ModelInfo(
                            modelId=item["id"],
                            sha=None,
                            sibling=None,
                            tags=None,
                            pipeline_tag=None,
                            author=None,
                            lastModified=None,
                            private=None,
                            gated=None,
                            downloads=None,
                            likes=None,
                            cardData=None,
                            modelIdFull=None,
                        )
                        for item in items
                    ]

                if not models:
                    raise ValueError(
                        f"No repositories match '{query}'. Try the full org/repo name."
                    )

                self.signals.combo_ready.emit(models)
            except Exception as e:
                self.signals.error.emit(str(e))
                self.repo_label.setText("Repository: —")
            finally:
                self.signals.finished.emit()

        threading.Thread(target=_search, daemon=True).start()

    # ------------------------------------------------------------------
    # Slot: populate the combo box with search results
    # ------------------------------------------------------------------
    def _on_combo_ready(self, models: List[ModelInfo]):
        self.match_combo.addItem("Select a repository …")
        for m in models:
            self.match_combo.addItem(m.modelId)
        self.match_combo.setCurrentIndex(0)
        self.match_combo.setVisible(True)
        self.repo_label.setText("Repository: (choose from list)")

    def _combo_selected(self, idx: int):
        if idx <= 0:  # placeholder entry
            return
        repo_id = self.match_combo.itemText(idx)
        self._finalize_repo_selection(repo_id)

    # ------------------------------------------------------------------
    # After we know the exact repo id, list its files (with accurate sizes)
    # ------------------------------------------------------------------
    def _finalize_repo_selection(self, repo_id: str):
        self.repo_label.setText(f"Repository: {repo_id}")
        self.download_repo_btn.setEnabled(True)

        def _list_files():
            try:
                raw_files = list_repo_files(repo_id=repo_id, revision="main")
                files: List[Dict] = []

                if raw_files and isinstance(raw_files[0], dict):
                    # Newer hub versions already give us size (if present)
                    for entry in raw_files:
                        if entry.get("type") != "file":
                            continue
                        size = entry.get("size")
                        if not size:
                            url = f"https://huggingface.co/{repo_id}/resolve/main/{entry['path']}"
                            size = self._cached_head_size(url)
                        files.append({"name": entry["path"], "size": size})
                else:
                    # Older hub versions: plain list of filenames.
                    urls = [
                        f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
                        for fn in raw_files
                    ]

                    # Determine which URLs we still need to query
                    uncached = [u for u in urls if u not in self._size_cache]

                    if uncached:
                        fetched = fetch_sizes_parallel(uncached, max_workers=8)
                        self._size_cache.update(fetched)

                    for fn, url in zip(raw_files, urls):
                        files.append(
                            {"name": fn, "size": self._size_cache.get(url, 0)}
                        )

                self.signals.files_ready.emit(files)
            except Exception as e:
                # If anything goes wrong we still emit an empty list so the UI
                # can recover gracefully.
                self.signals.error.emit(f"Failed to list files: {e}")
                self.signals.files_ready.emit([])
            finally:
                self.signals.finished.emit()

        threading.Thread(target=_list_files, daemon=True).start()

    # ------------------------------------------------------------------
    # Small wrapper that checks the in‑memory cache before issuing a HEAD
    # ------------------------------------------------------------------
    def _cached_head_size(self, url: str) -> int:
        if url in self._size_cache:
            return self._size_cache[url]
        sz = fetch_remote_size(url)
        self._size_cache[url] = sz
        return sz

    # ------------------------------------------------------------------
    # Slot: populate the file table widget (center size column)
    # ------------------------------------------------------------------
    def _on_files_ready(self, files: List[Dict]):
        self.file_table.setRowCount(0)
        total = 0
        for row, f in enumerate(files):
            self.file_table.insertRow(row)

            name_item = QTableWidgetItem(f["name"])
            size_item = QTableWidgetItem(self._human_readable(f["size"]))
            size_item.setTextAlignment(Qt.AlignCenter)   # center the size value

            self.file_table.setItem(row, 0, name_item)
            self.file_table.setItem(row, 1, size_item)

            total += f["size"]

        self.repo_files = files
        self.repo_total_bytes = total
        self.repo_downloaded_bytes = 0

        self.overall_label.setText(
            f"Total repository size: {self._human_readable(total)}"
        )
        self.file_table.setEnabled(True)
        self.download_file_btn.setEnabled(True)

        # Resize the size column now that we know the longest string
        self.file_table.resizeColumnToContents(1)

    # ------------------------------------------------------------------
    # Helper: turn a byte count into a friendly string
    # ------------------------------------------------------------------
    @staticmethod
    def _human_readable(num_bytes: int) -> str:
        """Return a string like '12.3 MiB'."""
        if num_bytes < 1024:
            return f"{num_bytes} B"
        for unit in ["KiB", "MiB", "GiB", "TiB"]:
            num_bytes /= 1024.0
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
        return f"{num_bytes:.2f} PiB"

    # ------------------------------------------------------------------
    # Download a single file (with progress bar)
    # ------------------------------------------------------------------
    def download_selected_files(self):
        selected_items = self.file_table.selectedItems()
        if not selected_items:
            self.show_error("Select at least one file from the table.")
            return

        # Get unique selected rows
        selected_rows = set()
        for item in selected_items:
            selected_rows.add(item.row())

        if not selected_rows:
            self.show_error("No rows selected.")
            return

        repo_id = self.repo_label.text().split(":")[1].strip()
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Select destination folder", str(Path.home())
        )
        if not dest_dir:
            return

        # Create base directory: dest_dir/repo_id
        base_dir = Path(dest_dir) / repo_id
        base_dir.mkdir(parents=True, exist_ok=True)

        filenames = []
        for row in selected_rows:
            filename_item = self.file_table.item(row, 0)
            if filename_item:
                filenames.append(filename_item.text())

        if not filenames:
            self.show_error("No files selected.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        def _download_files():
            try:
                total_files = len(filenames)
                for idx, filename in enumerate(filenames, start=1):
                    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                    dest_path = base_dir / filename

                    # Ensure the parent directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    self.signals.progress.emit(
                        f"Downloading {filename} ({idx}/{total_files})"
                    )
                    stream_download(file_url, dest_path, self.signals)

                self.signals.progress.emit(f"Downloaded {total_files} file(s) to {base_dir}")
            except Exception as e:
                self.signals.error.emit(str(e))
            finally:
                self.progress_bar.setVisible(False)
                self.signals.finished.emit()

        threading.Thread(target=_download_files, daemon=True).start()

    # ------------------------------------------------------------------
    # Download the whole repository (file‑by‑file, overall progress)
    # ------------------------------------------------------------------
    def download_entire_repo(self):
        repo_id = self.repo_label.text().split(":")[1].strip()
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Select folder to store the repository", str(Path.home())
        )
        if not dest_dir:
            return

        # Show overall progress widgets
        self.overall_progress_bar.setVisible(True)
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.update_overall_status(
            f"0 MiB / {self._human_readable(self.repo_total_bytes)}"
        )

        def _download_repo():
            try:
                base_dir = Path(dest_dir) / repo_id
                base_dir.mkdir(parents=True, exist_ok=True)

                total_files = len(self.repo_files)
                for idx, file_info in enumerate(self.repo_files, start=1):
                    file_name = file_info["name"]
                    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"
                    dest_path = base_dir / Path(file_name)

                    self.signals.progress.emit(
                        f"Downloading {file_name} ({idx}/{total_files})"
                    )
                    file_bytes = stream_download(file_url, dest_path, self.signals)

                    # ---- Update overall counters ----
                    self.repo_downloaded_bytes += file_bytes
                    overall_pct = int(
                        self.repo_downloaded_bytes * 100 / self.repo_total_bytes
                    )
                    self.signals.overall_progress_percent.emit(overall_pct)
                    self.signals.overall_progress.emit(
                        f"{self._human_readable(self.repo_downloaded_bytes)} / "
                        f"{self._human_readable(self.repo_total_bytes)}"
                    )

                self.signals.progress.emit("Repository download complete")
            except Exception as e:
                self.signals.error.emit(str(e))
            finally:
                self.overall_progress_bar.setVisible(False)
                self.signals.finished.emit()

        threading.Thread(target=_download_repo, daemon=True).start()


# ----------------------------------------------------------------------
# Application entry point – high‑DPI handling is automatic in Qt 6
# ----------------------------------------------------------------------
def main():
    # In Qt 6 high‑DPI scaling is enabled by default, so we omit the
    # deprecated Qt.AA_EnableHighDpiScaling / Qt.AA_UseHighDpiPixmaps.
    app = QApplication(sys.argv)

    # On macOS Qt automatically follows the system appearance (dark/light).
    # No forced palette is applied – the UI will look native.
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
