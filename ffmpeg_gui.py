
#!/usr/bin/env python3
"""
ffmpeg_filter_gui.py

Single-file Python GUI for building and running ffmpeg filtergraphs.
Features:
- Modern PySide6 GUI
- Drop-down filter selector with parameter widgets generated dynamically
- Filter chain visual editor (add/remove/reorder)
- Command preview and copy-to-clipboard
- Runs ffmpeg in background thread and parses progress using ffprobe
- Preview frame extraction using ffmpeg (shows still frame)
- Save/load presets (JSON)
- Basic validation and error handling
- Batch processing with multiple output formats

Dependencies: PySide6 (pip install PySide6)
Requires ffmpeg and ffprobe available on PATH.

Designed for advanced customization and reliability, not a full replacement for manual ffmpeg usage.

"""

import sys
import os
import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QLineEdit, QTextEdit, QListWidget,
    QListWidgetItem, QMessageBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QSplitter, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QAbstractItemView
)
from PySide6.QtGui import QPixmap, QClipboard, QPalette, QDragEnterEvent, QDropEvent

# --- Filter metadata ---
FILTER_DEFS: Dict[str, Dict[str, Any]] = {
    "scale": {
        "description": "Resize video, example: scale=1280:720",
        "params": {
            "width": {"type": "str", "default": "1920", "hint": "width or -1"},
            "height": {"type": "str", "default": "1080", "hint": "height or -1"},
            "flags": {"type": "str", "default": "lanczos", "hint": "scaling flag"}
        }
    },
    "crop": {
        "description": "Crop rectangle: crop=w:h:x:y",
        "params": {
            "w": {"type": "str", "default": "iw/2", "hint": "width expression"},
            "h": {"type": "str", "default": "ih/2", "hint": "height expression"},
            "x": {"type": "str", "default": "(iw-w)/2", "hint": "x offset"},
            "y": {"type": "str", "default": "(ih-h)/2", "hint": "y offset"}
        }
    },
    "eq": {
        "description": "Adjust brightness/contrast/saturation/gamma",
        "params": {
            "brightness": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0},
            "contrast": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
            "saturation": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
            "gamma": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0}
        }
    },
    "hue": {
        "description": "Adjust hue and saturation",
        "params": {
            "h": {"type": "int", "default": 0, "min": -180, "max": 180},
            "s": {"type": "int", "default": 0, "min": -100, "max": 100}
        }
    },
    "drawtext": {
        "description": "Overlay text onto video",
        "params": {
            "text": {"type": "str", "default": "Sample Text"},
            "fontfile": {"type": "str", "default": "", "hint": "path to ttf"},
            "fontsize": {"type": "int", "default": 24},
            "fontcolor": {"type": "str", "default": "white"},
            "x": {"type": "str", "default": "10"},
            "y": {"type": "str", "default": "10"}
        }
    },
    "fps": {
        "description": "Force frames per second",
        "params": {
            "fps": {"type": "int", "default": 30, "min": 1, "max": 240}
        }
    }
}

# --- Output format presets ---
OUTPUT_FORMATS: Dict[str, Dict[str, Any]] = {
    "MP4 (H.264)": {
        "extension": ".mp4",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "extra_args": ["-movflags", "+faststart"]
    },
    "MP4 (H.265/HEVC)": {
        "extension": ".mp4",
        "video_codec": "libx265",
        "audio_codec": "aac",
        "extra_args": ["-movflags", "+faststart"]
    },
    "MKV (H.264)": {
        "extension": ".mkv",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "extra_args": []
    },
    "WebM (VP9)": {
        "extension": ".webm",
        "video_codec": "libvpx-vp9",
        "audio_codec": "libopus",
        "extra_args": ["-deadline", "good", "-cpu-used", "2"]
    },
    "AVI (MPEG-4)": {
        "extension": ".avi",
        "video_codec": "mpeg4",
        "audio_codec": "mp3",
        "extra_args": []
    },
    "MOV (ProRes)": {
        "extension": ".mov",
        "video_codec": "prores_ks",
        "audio_codec": "pcm_s16le",
        "extra_args": ["-profile:v", "2"]
    },
    "Copy Codecs": {
        "extension": ".mp4",
        "video_codec": "copy",
        "audio_codec": "copy",
        "extra_args": []
    }
}

# --- Batch processing status ---
class BatchStatus(Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SKIPPED = "Skipped"

# --- Helper dataclasses ---
@dataclass
class FilterInstance:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_filter_str(self) -> str:
        if not self.params:
            return self.name
        pairs = []
        for k, v in self.params.items():
            if isinstance(v, bool):
                val = "1" if v else "0"
            else:
                val = str(v)
            pairs.append(f"{k}={val}")
        return f"{self.name}=" + ":".join(pairs)

@dataclass
class BatchFileItem:
    input_path: str
    output_path: str
    format_preset: str
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    error_message: str = ""

# --- Command validation and fixing ---
def validate_and_fix_ffmpeg_command(cmd: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validates and fixes common ffmpeg command-line issues.
    Returns (fixed_cmd, warnings)
    """
    warnings = []
    fixed_cmd = cmd.copy()

    # Check for missing input/output files
    if "-i" not in fixed_cmd:
        warnings.append("Warning: No input file specified (-i).")
    if not any(not arg.startswith("-") for arg in fixed_cmd):
        warnings.append("Warning: No output file specified.")

    # Check for common filter errors
    if "-vf" in fixed_cmd:
        vf_index = fixed_cmd.index("-vf")
        if len(fixed_cmd) <= vf_index + 1 or fixed_cmd[vf_index + 1].startswith("-"):
            warnings.append("Warning: No filter specified after -vf. Using 'null' filter.")
            fixed_cmd.insert(vf_index + 1, "null")

    # Check for missing codec arguments
    if "-c:v" not in fixed_cmd and "-c:a" not in fixed_cmd:
        warnings.append("Warning: No video/audio codec specified. Using default (libx264 for video).")
        fixed_cmd.extend(["-c:v", "libx264"])

    # Check for invalid filter syntax (e.g., missing '=' or ':')
    if "-vf" in fixed_cmd:
        vf_index = fixed_cmd.index("-vf")
        filter_str = fixed_cmd[vf_index + 1]
        if "=" not in filter_str and ":" not in filter_str:
            warnings.append(f"Warning: Invalid filter syntax: '{filter_str}'. Using 'null' filter.")
            fixed_cmd[vf_index + 1] = "null"

    # Check for unsupported codecs
    video_codec_index = fixed_cmd.index("-c:v") + 1 if "-c:v" in fixed_cmd else None
    if video_codec_index and fixed_cmd[video_codec_index] not in ["libx264", "libx265", "copy"]:
        warnings.append(f"Warning: Unsupported video codec '{fixed_cmd[video_codec_index]}'. Using 'libx264'.")
        fixed_cmd[video_codec_index] = "libx264"

    return fixed_cmd, warnings

# --- FFmpegWorker and utility functions ---
class FFmpegWorker(QThread):
    progress = Signal(float)
    log = Signal(str)
    finished_signal = Signal(int)

    def __init__(self, cmd: List[str], input_duration: Optional[float]):
        super().__init__()
        self.cmd = cmd
        self.input_duration = input_duration
        self._proc = None

    def run(self):
        try:
            self.log.emit("Running: " + " ".join(self.cmd))
            self._proc = subprocess.Popen(self.cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            while True:
                line = self._proc.stderr.readline()
                if not line:
                    break
                self.log.emit(line.strip())
                if "time=" in line and self.input_duration:
                    try:
                        t = self._parse_time_from_line(line)
                        if t is not None:
                            frac = min(max(t / self.input_duration, 0.0), 1.0)
                            self.progress.emit(frac)
                    except Exception:
                        pass
            self._proc.wait()
            code = self._proc.returncode
            self.progress.emit(1.0)
            self.finished_signal.emit(code)
        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished_signal.emit(-1)

    def _parse_time_from_line(self, line: str) -> Optional[float]:
        parts = line.split()
        for p in parts:
            if p.startswith("time="):
                t = p.split("=", 1)[1].strip().rstrip(',')
                try:
                    h, m, s = t.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(s)
                except Exception:
                    return None
        return None

# --- Batch processing worker ---
class BatchWorker(QThread):
    file_started = Signal(int)  # file index
    file_progress = Signal(int, float)  # file index, progress (0-1)
    file_completed = Signal(int, bool, str)  # file index, success, error_message
    batch_completed = Signal(int, int)  # successful count, failed count
    log = Signal(str)

    def __init__(self, batch_items: List[BatchFileItem], filter_chain: List[FilterInstance],
                 extra_args: List[str], quality: int):
        super().__init__()
        self.batch_items = batch_items
        self.filter_chain = filter_chain
        self.extra_args = extra_args
        self.quality = quality
        self._should_stop = False

    def run(self):
        successful = 0
        failed = 0

        for idx, item in enumerate(self.batch_items):
            if self._should_stop:
                self.log.emit(f"Batch processing cancelled by user")
                break

            self.file_started.emit(idx)
            self.log.emit(f"Processing: {os.path.basename(item.input_path)}")

            try:
                # Build command for this file
                cmd = build_ffmpeg_command(
                    item.input_path,
                    item.output_path,
                    self.filter_chain,
                    self.extra_args,
                    self.quality,
                    item.format_preset
                )

                # Get duration for progress tracking
                duration = probe_duration(item.input_path)

                # Run ffmpeg
                self.log.emit(f"Command: {' '.join(cmd)}")
                proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

                while True:
                    if self._should_stop:
                        proc.terminate()
                        self.log.emit(f"File processing cancelled: {os.path.basename(item.input_path)}")
                        break

                    line = proc.stderr.readline()
                    if not line:
                        break

                    # Parse progress
                    if "time=" in line and duration:
                        try:
                            t = self._parse_time_from_line(line)
                            if t is not None:
                                frac = min(max(t / duration, 0.0), 1.0)
                                self.file_progress.emit(idx, frac)
                        except Exception:
                            pass

                proc.wait()
                returncode = proc.returncode

                if returncode == 0:
                    self.log.emit(f"Completed: {os.path.basename(item.output_path)}")
                    self.file_completed.emit(idx, True, "")
                    successful += 1
                else:
                    error_msg = f"FFmpeg exited with code {returncode}"
                    self.log.emit(f"Failed: {os.path.basename(item.input_path)} - {error_msg}")
                    self.file_completed.emit(idx, False, error_msg)
                    failed += 1

            except Exception as e:
                error_msg = str(e)
                self.log.emit(f"Error processing {os.path.basename(item.input_path)}: {error_msg}")
                self.file_completed.emit(idx, False, error_msg)
                failed += 1

        self.batch_completed.emit(successful, failed)

    def _parse_time_from_line(self, line: str) -> Optional[float]:
        parts = line.split()
        for p in parts:
            if p.startswith("time="):
                t = p.split("=", 1)[1].strip().rstrip(',')
                try:
                    h, m, s = t.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(s)
                except Exception:
                    return None
        return None

    def stop(self):
        self._should_stop = True

def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def probe_duration(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(out.strip())
    except Exception:
        return None

def build_ffmpeg_command(input_file: str, output_file: str, filter_chain: List[FilterInstance],
                         extra_args: List[str], quality: int, format_preset: Optional[str] = None) -> List[str]:
    filters = ",".join([f.to_filter_str() for f in filter_chain]) if filter_chain else None
    cmd = ["ffmpeg", "-y", "-i", input_file]
    if filters:
        cmd += ["-vf", filters]

    # Handle format preset if provided (for batch processing)
    if format_preset and format_preset in OUTPUT_FORMATS:
        preset = OUTPUT_FORMATS[format_preset]
        v_codec = preset.get("video_codec", "libx264")
        a_codec = preset.get("audio_codec", "aac")
        preset_extra = preset.get("extra_args", [])

        # For copy codec, skip quality settings
        if v_codec == "copy":
            cmd += ["-c:v", "copy", "-c:a", "copy"]
        else:
            # Add quality-based encoding
            if quality >= 80:
                cmd += ["-c:v", v_codec, "-crf", "18", "-preset", "slow"]
            elif quality >= 60:
                cmd += ["-c:v", v_codec, "-crf", "23", "-preset", "medium"]
            elif quality >= 40:
                cmd += ["-c:v", v_codec, "-crf", "28", "-preset", "fast"]
            else:
                cmd += ["-c:v", v_codec, "-crf", "32", "-preset", "veryfast"]

            cmd += ["-c:a", a_codec, "-strict", "-2", "-b:a", "128k"]

        cmd += preset_extra
        cmd += extra_args
    else:
        # Original single-file mode logic
        if any(arg.startswith(('-c:v', '-vcodec')) for arg in extra_args):
            # User specified video codec, don't override
            cmd += extra_args
        else:
            # Add quality-based encoding
            if quality >= 80:
                cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "slow"]
            elif quality >= 60:
                cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]
            elif quality >= 40:
                cmd += ["-c:v", "libx264", "-crf", "28", "-preset", "fast"]
            else:
                cmd += ["-c:v", "libx264", "-crf", "32", "-preset", "veryfast"]

            # Add audio encoding if not specified
            if not any(arg.startswith(('-c:a', '-acodec')) for arg in extra_args):
                cmd += ["-c:a", "aac", "-strict", "-2", "-b:a", "128k"]

            # Add any other extra args
            cmd += [arg for arg in extra_args if not arg.startswith(('-c:v', '-vcodec', '-c:a', '-acodec'))]

    cmd += [output_file]
    fixed_cmd, warnings = validate_and_fix_ffmpeg_command(cmd)
    for warning in warnings:
        print(f"Command Warning: {warning}")
    return fixed_cmd

# --- GUI components ---
class FilterEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.filter_select = QComboBox()
        self.filter_select.addItems(sorted(FILTER_DEFS.keys()))
        self.layout.addWidget(QLabel("Choose filter"))
        self.layout.addWidget(self.filter_select)
        self.params_form = QFormLayout()
        self.params_widget = QWidget()
        self.params_widget.setLayout(self.params_form)
        self.layout.addWidget(self.params_widget)
        self.add_btn = QPushButton("Add to chain")
        self.layout.addWidget(self.add_btn)
        self.current_param_widgets: Dict[str, QWidget] = {}
        self.filter_select.currentTextChanged.connect(self._on_filter_change)
        self._on_filter_change(self.filter_select.currentText())

    def _clear_form(self):
        for i in reversed(range(self.params_form.count())):
            item = self.params_form.itemAt(i)
            if item:
                w = item.widget()
                if w:
                    w.deleteLater()
        self.current_param_widgets.clear()

    def _on_filter_change(self, name: str):
        self._clear_form()
        meta = FILTER_DEFS.get(name, {})
        params = meta.get("params", {})
        for pname, pmeta in params.items():
            ptype = pmeta.get("type", "str")
            default = pmeta.get("default", "")
            hint = pmeta.get("hint", "")
            if ptype == "int":
                w = QSpinBox()
                w.setRange(pmeta.get("min", -999999), pmeta.get("max", 999999))
                w.setValue(int(default))
            elif ptype == "float":
                w = QDoubleSpinBox()
                w.setRange(pmeta.get("min", -999999.0), pmeta.get("max", 999999.0))
                w.setDecimals(3)
                w.setValue(float(default))
            elif ptype == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            else:
                w = QLineEdit()
                w.setText(str(default))
                if hint:
                    w.setPlaceholderText(hint)
            self.params_form.addRow(QLabel(pname), w)
            self.current_param_widgets[pname] = w

    def get_filter_instance(self) -> FilterInstance:
        name = self.filter_select.currentText()
        params = {}
        for k, w in self.current_param_widgets.items():
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox):
                params[k] = w.value()
            elif isinstance(w, QCheckBox):
                params[k] = w.isChecked()
            else:
                text = w.text().strip()
                params[k] = text
        return FilterInstance(name=name, params=params)

class BatchProcessingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_items: List[BatchFileItem] = []
        self.batch_worker: Optional[BatchWorker] = None

        layout = QVBoxLayout(self)

        # Top controls
        controls_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        controls_layout.addWidget(self.add_files_btn)
        controls_layout.addWidget(self.remove_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Output directory and format
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Leave empty to use input directory with '_converted' suffix")
        self.output_dir.textChanged.connect(self.on_output_settings_changed)
        output_layout.addWidget(self.output_dir)
        self.browse_dir_btn = QPushButton("Browse")
        self.browse_dir_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_dir_btn)

        output_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(list(OUTPUT_FORMATS.keys()))
        self.format_combo.currentTextChanged.connect(self.on_output_settings_changed)
        output_layout.addWidget(self.format_combo)

        self.update_paths_btn = QPushButton("Update Paths")
        self.update_paths_btn.clicked.connect(self.on_output_settings_changed)
        self.update_paths_btn.setToolTip("Manually update output paths for all pending files")
        output_layout.addWidget(self.update_paths_btn)

        layout.addLayout(output_layout)

        # File list table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Input File", "Output File", "Format", "Status", "Progress"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 150)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Bottom controls
        bottom_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton("Start Batch Processing")
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        self.stop_batch_btn = QPushButton("Stop")
        self.stop_batch_btn.setEnabled(False)
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        bottom_layout.addWidget(self.start_batch_btn)
        bottom_layout.addWidget(self.stop_batch_btn)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

        # Log
        layout.addWidget(QLabel("Batch Log:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        layout.addWidget(self.log_view)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select video files")
        if not files:
            return

        output_dir = self.output_dir.text().strip()
        format_preset = self.format_combo.currentText()
        extension = OUTPUT_FORMATS[format_preset]["extension"]

        for file_path in files:
            input_path_obj = Path(file_path)

            # Generate output path
            if output_dir:
                # Use specified output directory
                output_path = os.path.join(output_dir, input_path_obj.stem + extension)
            else:
                # No output directory specified - use same directory as input
                # But add "_converted" suffix to avoid overwriting the original
                if input_path_obj.suffix.lower() == extension.lower():
                    # Same extension - add suffix to filename
                    output_path = str(input_path_obj.parent / (input_path_obj.stem + "_converted" + extension))
                else:
                    # Different extension - just change extension
                    output_path = str(input_path_obj.with_suffix(extension))

            # Safety check: ensure output path is never same as input path
            if os.path.normpath(output_path).lower() == os.path.normpath(file_path).lower():
                output_path = str(input_path_obj.parent / (input_path_obj.stem + "_converted" + extension))

            # Add to batch items
            item = BatchFileItem(
                input_path=file_path,
                output_path=output_path,
                format_preset=format_preset
            )
            self.batch_items.append(item)

        self.refresh_table()

    def remove_selected(self):
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        for row in selected_rows:
            if 0 <= row < len(self.batch_items):
                del self.batch_items[row]
        self.refresh_table()

    def clear_all(self):
        self.batch_items.clear()
        self.refresh_table()

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if dir_path:
            self.output_dir.setText(dir_path)

    def on_output_settings_changed(self):
        """Update output paths for all pending items when format or output directory changes"""
        if not self.batch_items:
            return

        format_preset = self.format_combo.currentText()
        if format_preset not in OUTPUT_FORMATS:
            return

        extension = OUTPUT_FORMATS[format_preset]["extension"]
        output_dir = self.output_dir.text().strip()

        updated_count = 0
        for item in self.batch_items:
            # Only update pending items (not processing/completed/failed)
            if item.status == BatchStatus.PENDING:
                item.format_preset = format_preset
                input_path_obj = Path(item.input_path)

                # Update output path with new settings
                if output_dir:
                    # Use specified output directory
                    item.output_path = os.path.join(output_dir, input_path_obj.stem + extension)
                else:
                    # No output directory - add suffix if same extension
                    if input_path_obj.suffix.lower() == extension.lower():
                        item.output_path = str(input_path_obj.parent / (input_path_obj.stem + "_converted" + extension))
                    else:
                        item.output_path = str(input_path_obj.with_suffix(extension))

                # Safety check: ensure output never overwrites input
                if os.path.normpath(item.output_path).lower() == os.path.normpath(item.input_path).lower():
                    item.output_path = str(input_path_obj.parent / (input_path_obj.stem + "_converted" + extension))

                updated_count += 1

        if updated_count > 0:
            self.refresh_table()
            self.log_view.append(f"Updated output paths for {updated_count} file(s)")

    def refresh_table(self):
        self.table.setRowCount(len(self.batch_items))
        for row, item in enumerate(self.batch_items):
            # Input file
            input_item = QTableWidgetItem(os.path.basename(item.input_path))
            input_item.setToolTip(item.input_path)  # Show full path on hover
            self.table.setItem(row, 0, input_item)

            # Output file
            output_item = QTableWidgetItem(os.path.basename(item.output_path))
            output_item.setToolTip(item.output_path)  # Show full path on hover
            self.table.setItem(row, 1, output_item)

            # Format
            format_item = QTableWidgetItem(item.format_preset)
            self.table.setItem(row, 2, format_item)

            # Status
            status_item = QTableWidgetItem(item.status.value)
            if item.error_message:
                status_item.setToolTip(item.error_message)  # Show error on hover
            self.table.setItem(row, 3, status_item)

            # Progress
            progress_bar = QProgressBar()
            progress_bar.setValue(int(item.progress * 100))
            self.table.setCellWidget(row, 4, progress_bar)

    def start_batch_processing(self):
        if not self.batch_items:
            QMessageBox.warning(self, "No Files", "Please add files to process first.")
            return

        # Validate that no output file will overwrite an input file
        conflicts = []
        for item in self.batch_items:
            if os.path.normpath(item.output_path).lower() == os.path.normpath(item.input_path).lower():
                conflicts.append(os.path.basename(item.input_path))

        if conflicts:
            QMessageBox.critical(
                self,
                "Output Path Conflict",
                f"The following files have output paths identical to input paths:\n\n" +
                "\n".join(conflicts) +
                "\n\nThis would overwrite the original files. Please specify an output directory or change the format."
            )
            return

        # Get filter chain and settings from main window
        main_window = self.window()
        if isinstance(main_window, MainWindow):
            filter_chain = main_window.get_chain()
            extra_args = main_window.extra_args.text().strip().split() if main_window.extra_args.text().strip() else []
            quality = main_window.quality_slider.value()
        else:
            filter_chain = []
            extra_args = []
            quality = 70

        # Reset all items to pending
        for item in self.batch_items:
            item.status = BatchStatus.PENDING
            item.progress = 0.0
            item.error_message = ""

        self.refresh_table()
        self.log_view.clear()

        # Start batch worker
        self.batch_worker = BatchWorker(self.batch_items, filter_chain, extra_args, quality)
        self.batch_worker.file_started.connect(self.on_file_started)
        self.batch_worker.file_progress.connect(self.on_file_progress)
        self.batch_worker.file_completed.connect(self.on_file_completed)
        self.batch_worker.batch_completed.connect(self.on_batch_completed)
        self.batch_worker.log.connect(self.on_log)
        self.batch_worker.start()

        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.add_files_btn.setEnabled(False)

    def stop_batch_processing(self):
        if self.batch_worker:
            self.batch_worker.stop()
            self.log_view.append("Stopping batch processing...")

    @Slot(int)
    def on_file_started(self, idx: int):
        if 0 <= idx < len(self.batch_items):
            self.batch_items[idx].status = BatchStatus.PROCESSING
            self.refresh_table()

    @Slot(int, float)
    def on_file_progress(self, idx: int, progress: float):
        if 0 <= idx < len(self.batch_items):
            self.batch_items[idx].progress = progress
            # Update just the progress bar
            progress_bar = self.table.cellWidget(idx, 4)
            if isinstance(progress_bar, QProgressBar):
                progress_bar.setValue(int(progress * 100))

    @Slot(int, bool, str)
    def on_file_completed(self, idx: int, success: bool, error_msg: str):
        if 0 <= idx < len(self.batch_items):
            if success:
                self.batch_items[idx].status = BatchStatus.COMPLETED
                self.batch_items[idx].progress = 1.0
            else:
                self.batch_items[idx].status = BatchStatus.FAILED
                self.batch_items[idx].error_message = error_msg
            self.refresh_table()

    @Slot(int, int)
    def on_batch_completed(self, successful: int, failed: int):
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)

        summary = f"\n=== Batch Processing Complete ===\n"
        summary += f"Successful: {successful}\n"
        summary += f"Failed: {failed}\n"
        summary += f"Total: {successful + failed}\n"
        self.log_view.append(summary)

        # Find first successful output file to determine output directory
        output_dir = None
        if successful > 0:
            for item in self.batch_items:
                if item.status == BatchStatus.COMPLETED and os.path.exists(item.output_path):
                    output_dir = os.path.dirname(item.output_path)
                    break

        # Ask user if they want to open output directory
        if output_dir:
            reply = QMessageBox.question(
                self,
                "Batch Processing Complete",
                summary + "\nWould you like to open the output directory?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                main_window = self.window()
                if isinstance(main_window, MainWindow):
                    main_window._open_directory(output_dir)
        else:
            QMessageBox.information(self, "Batch Complete", summary)

    @Slot(str)
    def on_log(self, message: str):
        self.log_view.append(message)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFmpeg FilterLab - Single & Batch Processing")
        self.resize(1200, 800)

        # Set window icon
        icon_path = self._get_resource_path("icon.ico")
        if os.path.exists(icon_path):
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

        self.container = QWidget()
        self.setCentralWidget(self.container)
        self.main_layout = QVBoxLayout(self.container)

        # --- Shared Quality slider ---
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(70)
        self.quality_slider.valueChanged.connect(self.update_command_preview)
        quality_layout.addWidget(self.quality_slider)
        self.quality_label = QLabel("High (CRF 18)")
        quality_layout.addWidget(self.quality_label)
        self.main_layout.addLayout(quality_layout)

        # --- Tab widget for Single/Batch modes ---
        self.mode_tabs = QTabWidget()
        self.main_layout.addWidget(self.mode_tabs)

        # === Single File Mode Tab ===
        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)

        # Top controls for single mode
        top = QHBoxLayout()
        self.input_path = QLineEdit()
        self.in_btn = QPushButton("Open input")
        self.in_btn.clicked.connect(self.open_input)
        self.output_path = QLineEdit()
        self.out_btn = QPushButton("Choose output")
        self.out_btn.clicked.connect(self.choose_output)
        top.addWidget(QLabel("Input:"))
        top.addWidget(self.input_path)
        top.addWidget(self.in_btn)
        top.addWidget(QLabel("Output:"))
        top.addWidget(self.output_path)
        top.addWidget(self.out_btn)
        single_layout.addLayout(top)

        # --- Split area: left editor, right preview and run ---
        split = QSplitter(Qt.Horizontal)

        # --- Left: filter editor and chain ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.filter_editor = FilterEditor()
        left_layout.addWidget(self.filter_editor)
        chain_label_layout = QHBoxLayout()
        chain_label_layout.addWidget(QLabel("Filter chain"))
        self.save_preset_btn = QPushButton("Save preset")
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.load_preset_btn = QPushButton("Load preset")
        self.load_preset_btn.clicked.connect(self.load_preset)
        chain_label_layout.addWidget(self.save_preset_btn)
        chain_label_layout.addWidget(self.load_preset_btn)
        left_layout.addLayout(chain_label_layout)
        self.chain_list = QListWidget()
        left_layout.addWidget(self.chain_list)
        chain_actions = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_filter_to_chain)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(self.move_up)
        self.down_btn = QPushButton("Down")
        self.down_btn.clicked.connect(self.move_down)
        chain_actions.addWidget(self.add_btn)
        chain_actions.addWidget(self.remove_btn)
        chain_actions.addWidget(self.up_btn)
        chain_actions.addWidget(self.down_btn)
        left_layout.addLayout(chain_actions)

        # --- Quick extras ---
        extras = QHBoxLayout()
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Additional ffmpeg args, e.g. -c:v libx264 -crf 18")
        extras.addWidget(QLabel("Extra args:"))
        extras.addWidget(self.extra_args)
        left_layout.addLayout(extras)
        split.addWidget(left)

        # --- Right: preview, command, run ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_label = QLabel("Preview frame will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Box)
        self.preview_label.setMinimumHeight(240)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.preview_label)
        preview_controls = QHBoxLayout()
        self.preview_at = QLineEdit("00:00:01")
        self.preview_btn = QPushButton("Grab preview frame")
        self.preview_btn.clicked.connect(self.grab_preview)
        self.live_preview_btn = QPushButton("Live Preview")
        self.live_preview_btn.clicked.connect(self.live_preview)
        preview_controls.addWidget(QLabel("Preview time:"))
        preview_controls.addWidget(self.preview_at)
        preview_controls.addWidget(self.preview_btn)
        preview_controls.addWidget(self.live_preview_btn)
        right_layout.addLayout(preview_controls)

        # --- Command preview ---
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        right_layout.addWidget(QLabel("FFmpeg command preview"))
        right_layout.addWidget(self.cmd_preview)
        cmd_actions = QHBoxLayout()
        self.copy_cmd_btn = QPushButton("Copy command")
        self.copy_cmd_btn.clicked.connect(self.copy_command)
        self.fix_cmd_btn = QPushButton("Fix Command")
        self.fix_cmd_btn.clicked.connect(self.fix_command)
        self.run_btn = QPushButton("Run ffmpeg")
        self.run_btn.clicked.connect(self.run_ffmpeg)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.clicked.connect(self.abort)
        self.abort_btn.setEnabled(False)
        cmd_actions.addWidget(self.copy_cmd_btn)
        cmd_actions.addWidget(self.fix_cmd_btn)
        cmd_actions.addWidget(self.run_btn)
        cmd_actions.addWidget(self.abort_btn)
        right_layout.addLayout(cmd_actions)

        # --- Log and progress ---
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_view)
        split.addWidget(right)
        single_layout.addWidget(split)

        # Add single file tab
        self.mode_tabs.addTab(single_widget, "Single File")

        # === Batch Processing Mode Tab ===
        self.batch_widget = BatchProcessingWidget(self)
        self.mode_tabs.addTab(self.batch_widget, "Batch Processing")

        # --- State ---
        self.ffmpeg_thread: Optional[FFmpegWorker] = None

        # --- Wire up ---
        self.filter_editor.add_btn.clicked.connect(self.add_filter_to_chain)
        self.chain_list.model().rowsInserted.connect(lambda: self.update_command_preview())
        self.chain_list.model().rowsRemoved.connect(lambda: self.update_command_preview())
        self.quality_slider.valueChanged.connect(self.update_command_preview)
        self.quality_slider.valueChanged.connect(self.update_quality_label)
        self.update_command_preview()
        self.update_quality_label(self.quality_slider.value())
        if not check_ffmpeg_available():
            QMessageBox.critical(self, "Missing ffmpeg", "ffmpeg or ffprobe not found on PATH. Install them and restart.")

    # --- UI actions ---
    def open_input(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open input file")
        if p:
            self.input_path.setText(p)
            self.update_command_preview()

    def choose_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Choose output file")
        if p:
            self.output_path.setText(p)
            self.update_command_preview()

    def add_filter_to_chain(self):
        fi = self.filter_editor.get_filter_instance()
        item = QListWidgetItem(fi.to_filter_str())
        item.setData(Qt.UserRole, fi)
        self.chain_list.addItem(item)
        self.update_command_preview()

    def remove_selected(self):
        idxs = [i.row() for i in self.chain_list.selectedIndexes()]
        for i in sorted(idxs, reverse=True):
            self.chain_list.takeItem(i)
        self.update_command_preview()

    def move_up(self):
        cur = self.chain_list.currentRow()
        if cur > 0:
            item = self.chain_list.takeItem(cur)
            self.chain_list.insertItem(cur-1, item)
            self.chain_list.setCurrentRow(cur-1)
            self.update_command_preview()

    def move_down(self):
        cur = self.chain_list.currentRow()
        if cur < self.chain_list.count() - 1 and cur >= 0:
            item = self.chain_list.takeItem(cur)
            self.chain_list.insertItem(cur+1, item)
            self.chain_list.setCurrentRow(cur+1)
            self.update_command_preview()

    def get_chain(self) -> List[FilterInstance]:
        out = []
        for i in range(self.chain_list.count()):
            item = self.chain_list.item(i)
            fi: FilterInstance = item.data(Qt.UserRole)
            out.append(fi)
        return out

    def update_command_preview(self):
        inp = self.input_path.text().strip()
        outp = self.output_path.text().strip() or "output.mp4"
        chain = self.get_chain()
        extra = self.extra_args.text().strip().split()
        quality = self.quality_slider.value()
        if inp:
            cmd = build_ffmpeg_command(inp, outp, chain, extra, quality)
            self.cmd_preview.setPlainText(" ".join(cmd))
            self.update_quality_label(quality)
        else:
            self.cmd_preview.setPlainText("Provide input file to generate command preview.")

    def copy_command(self):
        cb: QClipboard = QApplication.clipboard()
        cb.setText(self.cmd_preview.toPlainText())

    def fix_command(self):
        inp = self.input_path.text().strip()
        outp = self.output_path.text().strip() or "output.mp4"
        chain = self.get_chain()
        extra = self.extra_args.text().strip().split()
        quality = self.quality_slider.value()
        cmd = build_ffmpeg_command(inp, outp, chain, extra, quality)
        fixed_cmd, warnings = validate_and_fix_ffmpeg_command(cmd)
        self.cmd_preview.setPlainText(" ".join(fixed_cmd))
        for warning in warnings:
            self.log_view.append(f"Fix: {warning}")

    def save_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save preset", filter="JSON files (*.json)")
        if not p:
            return
        data = {
            "chain": [ {"name": fi.name, "params": fi.params} for fi in self.get_chain() ],
            "extra_args": self.extra_args.text()
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Saved", "Preset saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load preset", filter="JSON files (*.json)")
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chain_list.clear()
            for fi in data.get("chain", []):
                inst = FilterInstance(name=fi["name"], params=fi.get("params", {}))
                item = QListWidgetItem(inst.to_filter_str())
                item.setData(Qt.UserRole, inst)
                self.chain_list.addItem(item)
            self.extra_args.setText(data.get("extra_args", ""))
            self.update_command_preview()
            QMessageBox.information(self, "Loaded", "Preset loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def grab_preview(self):
        inp = self.input_path.text().strip()
        if not inp or not os.path.exists(inp):
            QMessageBox.warning(self, "Missing input", "Provide a valid input file first.")
        t = self.preview_at.text().strip() or "00:00:01"
        tmp = os.path.join(tempfile.gettempdir(), f"ff_preview_{int(time.time())}.jpg")
        chain = self.get_chain()
        filters = ",".join([f.to_filter_str() for f in chain]) if chain else None
        cmd = ["ffmpeg", "-y", "-ss", t, "-i", inp, "-vframes", "1"]
        if filters:
            cmd += ["-vf", filters]
        cmd += [tmp]
        try:
            subprocess.check_call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            pix = QPixmap(tmp)
            if pix and not pix.isNull():
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.preview_label.setText("Failed to load preview image")
        except Exception as e:
            QMessageBox.critical(self, "Preview error", f"Failed to grab preview: {e}")

    def live_preview(self):
        inp = self.input_path.text().strip()
        if not inp or not os.path.exists(inp):
            QMessageBox.warning(self, "Missing input", "Provide a valid input file first.")
        chain = self.get_chain()
        filters = ",".join([f.to_filter_str() for f in chain]) if chain else None
        cmd = ["ffplay", "-i", inp]
        if filters:
            cmd += ["-vf", filters]
        try:
            subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to start live preview: {e}")

    def run_ffmpeg(self):
        inp = self.input_path.text().strip()
        outp = self.output_path.text().strip()
        if not inp or not os.path.exists(inp):
            QMessageBox.warning(self, "Missing input", "Provide a valid input file first.")
        
        if not outp:
            QMessageBox.warning(self, "Missing output", "Choose an output path.")
        
        chain = self.get_chain()
        extra = self.extra_args.text().strip().split() if self.extra_args.text().strip() else []
        quality = self.quality_slider.value()
        self.log_view.append("Building ffmpeg command...")
        cmd = build_ffmpeg_command(inp, outp, chain, extra, quality)
        self.log_view.append("Ffmpeg command: " + " ".join(cmd))
        self.log_view.append("Ffmpeg command built.")
        dur = probe_duration(inp)
        self.log_view.clear()
        self.log_view.append("Creating FFmpegWorker...")
        self.ffmpeg_thread = FFmpegWorker(cmd, dur)
        self.log_view.append("FFmpegWorker created.")
        self.ffmpeg_thread.progress.connect(self._on_progress)
        self.ffmpeg_thread.log.connect(self._on_log)
        self.ffmpeg_thread.finished_signal.connect(self._on_finished)
        self.ffmpeg_thread.start()
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

    @Slot(float)
    def _on_progress(self, v: float):
        pct = int(v * 100)
        self.statusBar().showMessage(f"Progress: {pct}%")

    @Slot(str)
    def _on_log(self, s: str):
        self.log_view.append(s)

    @Slot(int)
    def _on_finished(self, code: int):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        if code == 0:
            output_path = self.output_path.text().strip()

            # Ask user if they want to open output directory
            reply = QMessageBox.question(
                self,
                "Processing Complete",
                "FFmpeg finished successfully!\n\nWould you like to open the output directory?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes and output_path and os.path.exists(output_path):
                output_dir = os.path.dirname(output_path)
                self._open_directory(output_dir)
        else:
            QMessageBox.warning(self, "Finished", f"ffmpeg exited with code {code}")

    def abort(self):
        if self.ffmpeg_thread and self.ffmpeg_thread._proc:
            try:
                self.ffmpeg_thread._proc.terminate()
                self._on_log("Process terminated by user")
            except Exception:
                pass

    def _open_directory(self, directory: str):
        """Open file explorer to the specified directory"""
        try:
            if sys.platform == 'win32':
                os.startfile(directory)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', directory])
            else:  # linux variants
                subprocess.Popen(['xdg-open', directory])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open directory: {e}")

    def update_quality_label(self, value):
        if value >= 80:
            self.quality_label.setText("High (CRF 18)")
        elif value >= 60:
            self.quality_label.setText("Medium (CRF 23)")
        elif value >= 40:
            self.quality_label.setText("Low (CRF 28)")
        else:
            self.quality_label.setText("Very Low (CRF 32)")

    def _get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

# --- Run ---
def _get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def main():
    app = QApplication(sys.argv)

    # Load stylesheet
    stylesheet_path = _get_resource_path("style.css")
    try:
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Warning: Could not load stylesheet: {e}")

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()