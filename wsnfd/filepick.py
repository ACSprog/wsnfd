"""Choosing the Intel files from the interface: a native file-explorer dialog, uploads, and forgiving path handling."""
import os, sys, subprocess

# The dialog runs in its own process: tkinter wants the main thread, which Streamlit's script thread is not,
# and a failure (no display, no tkinter) must never take the app down.
_DIALOG = r'''
import sys, tkinter as tk
from tkinter import filedialog
title, patterns, start = sys.argv[1], sys.argv[2], sys.argv[3]
root = tk.Tk(); root.withdraw()
try: root.attributes("-topmost", True)
except Exception: pass
p = filedialog.askopenfilename(title=title, filetypes=[("Allowed files", tuple(patterns.split(";"))), ("All files", "*.*")], initialdir=start or None)
root.destroy(); sys.stdout.write(p or "")
'''

def pick_file(title, patterns="*.*", start="", _script=_DIALOG, timeout=600):
    """Open the operating system's file window. Returns (path or None, message). Only works on the computer running the app."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([sys.executable, "-c", _script, title, patterns, start], capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    except Exception as e:
        return None, f"The file window could not be opened ({e}). Paste the path or use the upload box instead."
    if r.returncode != 0:
        why = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        return None, f"The file window could not be opened ({why}). Paste the path or use the upload box instead."
    p = r.stdout.strip()
    return (p, "") if p else (None, "No file was chosen.")

def clean_path(p):
    """Tolerate quotes (Windows 'Copy as path'), stray spaces, ~ and %VARS%."""
    return os.path.expandvars(os.path.expanduser((p or "").strip().strip('"').strip("'").strip()))

def _find_in(folder, names, depth=2):
    want = {n.lower() for n in names}
    for root, dirs, files in os.walk(folder):
        if root[len(folder):].count(os.sep) >= depth: dirs[:] = []
        for f in sorted(files):
            if f.lower() in want: return os.path.join(root, f)
    return None

def resolve_intel(path):
    """Accept the data file or a folder that contains it. Returns (file path or None, message)."""
    p = clean_path(path)
    if not p: return None, "Choose the Intel data file with Browse, upload it, or paste its path."
    if os.path.isfile(p): return p, ""
    if os.path.isdir(p):
        f = _find_in(p, ["data.txt.gz", "data.txt"])
        return (f, f"Found {os.path.basename(f)} in that folder.") if f else (None, "That folder has no data.txt.gz or data.txt. Use Browse to pick the file.")
    return None, "That path does not exist. Use Browse to pick the file."

def resolve_locs(path, data_path=None):
    """The mote_locs.txt path if given and valid, else one sitting next to (or near) the data file. Returns (path or None, message)."""
    p = clean_path(path)
    if p:
        if os.path.isfile(p): return p, ""
        if os.path.isdir(p):
            f = _find_in(p, ["mote_locs.txt"]); return (f, "") if f else (None, "No mote_locs.txt in that folder.")
        return None, "That mote_locs.txt path does not exist."
    if data_path:
        f = _find_in(os.path.dirname(os.path.abspath(data_path)), ["mote_locs.txt"], depth=1)
        if f: return f, "Using the mote_locs.txt found next to the data file."
    return None, "No mote_locs.txt: nodes will be treated as unlocated (no spatial neighbours)."

def save_upload(uploaded, dest_dir):
    """Write a Streamlit UploadedFile to disk and return its path. The file is kept so it can be reused next time."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, os.path.basename(uploaded.name))
    with open(out, "wb") as f: f.write(uploaded.getbuffer())
    return out

def file_stamp(path):
    """(size, modified time) so a replaced file is not served from a stale cache."""
    try: s = os.stat(path); return (s.st_size, int(s.st_mtime))
    except (OSError, TypeError, ValueError): return (0, 0)      # missing file, or no path given (None)
