"""Modern flat dark theme + status colors for the recorder/replay GUIs."""

from __future__ import annotations

# semantic colors (GitHub-dark-ish)
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
OK = "#2ea043"
WARN = "#d29922"
BAD = "#f85149"
IDLE = "#30363d"

# status banner color per state
STATE_COLORS = {
    "IDLE": IDLE,
    "ARMED": ACCENT,
    "ENGAGED": OK,
    "REC": OK,
    "HOMING": WARN,
    "REVIEW": WARN,
    "ERROR": BAD,
}

QSS = f"""
* {{ font-family: -apple-system, "Segoe UI", "Noto Sans", sans-serif; font-size: 13px; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid #30363d; border-radius: 10px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 6px 8px; color: {TEXT};
}}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid #30363d; selection-background-color: {ACCENT}; }}
QPushButton {{
    background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 8px 14px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #30363d; }}
QPushButton:disabled {{ color: #484f58; }}
QPushButton:checked {{ background: {OK}; border-color: {OK}; color: white; }}
QSlider::groove:horizontal {{ height: 6px; background: #30363d; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QCheckBox {{ spacing: 6px; }}
"""


def banner_style(color: str) -> str:
    return f"background: {color}; color: white; border-radius: 10px; padding: 10px;font-size: 18px; font-weight: 600;"


def dot(ok: bool) -> str:
    """Inline HTML colored dot for the health strip."""
    return f'<span style="color:{OK if ok else BAD};">●</span>'
