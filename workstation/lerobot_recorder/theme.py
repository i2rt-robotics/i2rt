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
* {{ font-family: -apple-system, "Segoe UI", "Noto Sans", sans-serif; font-size: 20px; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid #30363d; border-radius: 10px; margin-top: 18px; padding: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; color: {TEXT}; font-size: 20px;
}}
QComboBox::drop-down {{ border: 0; width: 28px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid #30363d; selection-background-color: {ACCENT}; }}
QPushButton {{
    background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 13px 22px; color: {TEXT}; font-size: 20px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #30363d; }}
QPushButton:disabled {{ color: #484f58; }}
QPushButton:checked {{ background: {OK}; border-color: {OK}; color: white; }}
QPushButton#start {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-size: 28px; font-weight: 700; padding: 18px 32px; }}
QPushButton#start:hover {{ background: #6cb1ff; }}
QPushButton#estop {{ font-size: 20px; font-weight: 700; }}
QCheckBox {{ spacing: 8px; font-size: 20px; }}
"""


def banner_style(color: str) -> str:
    return f"background: {color}; color: white; border-radius: 10px; padding: 14px; font-size: 30px; font-weight: 600;"


def dot(ok: bool) -> str:
    """Inline HTML colored dot for the health strip."""
    return f'<span style="color:{OK if ok else BAD};">●</span>'
