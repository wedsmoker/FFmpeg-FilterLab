

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
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QLineEdit, QTextEdit, QListWidget,
    QListWidgetItem, QMessageBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QSplitter, QSizePolicy, QSlider
)
from PySide6.QtGui import QPixmap, QClipboard, QPalette

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
                         extra_args: List[str], quality: int) -> List[str]:
    filters = ",".join([f.to_filter_str() for f in filter_chain]) if filter_chain else None
    cmd = ["ffmpeg", "-y", "-i", input_file]
    if filters:
        cmd += ["-vf", filters]
    
    # Add quality/size control for video
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFmpeg FilterLab")
        self.resize(1100, 700)
        self.container = QWidget()
        self.setCentralWidget(self.container)
        self.main_layout = QVBoxLayout(self.container)

        # --- Top controls ---
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
        self.main_layout.addLayout(top)

        # --- Quality slider ---
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
        self.main_layout.addWidget(split)

        # --- State ---
        self.ffmpeg_thread: Optional[FFmpegWorker] = None

        # --- Wire up ---
        self.filter_editor.add_btn.clicked.connect(self.add_filter_to_chain)
        self.chain_list.model().rowsInserted.connect(lambda: self.update_command_preview())
        self.chain_list.model().rowsRemoved.connect(lambda: self.update_command_preview())
        self.quality_slider.valueChanged.connect(self.update_command_preview)
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
            QMessageBox.information(self, "Done", "ffmpeg finished successfully.")
        else:
            QMessageBox.warning(self, "Finished", f"ffmpeg exited with code {code}")

    def abort(self):
        if self.ffmpeg_thread and self.ffmpeg_thread._proc:
            try:
                self.ffmpeg_thread._proc.terminate()
                self._on_log("Process terminated by user")
            except Exception:
                pass

    def update_quality_label(self, value):
        if value >= 80:
            self.quality_label.setText("High (CRF 18)")
        elif value >= 60:
            self.quality_label.setText("Medium (CRF 23)")
        elif value >= 40:
            self.quality_label.setText("Low (CRF 28)")
        else:
            self.quality_label.setText("Very Low (CRF 32)")

# --- Run ---
def main():
    app = QApplication(sys.argv)
    # Load stylesheet
    if getattr(sys, 'frozen', False):
        # running in a bundle
        basedir = os.path.dirname(sys.executable)
    else:
        # running live
        basedir = os.path.dirname(os.path.abspath(__file__))
    

    if getattr(sys, 'frozen', False):
        # running in a bundle
        basedir = sys._MEIPASS
    stylesheet_path = os.path.join(basedir, "style.css")
    try:
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        pass


    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':

    main()
