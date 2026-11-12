"""Download the Intel Lab location file from its original source (runs on the user's own computer, which has internet)."""
import os, tempfile, urllib.request

LOCS_URL = "https://db.csail.mit.edu/labdata/mote_locs.txt"

def validate_locs(text, min_rows=10):
    """True if the text looks like 'moteid x y' lines, e.g. not an error page."""
    ok = 0
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 3:
            try: int(p[0]); float(p[1]); float(p[2]); ok += 1
            except ValueError: return False, "That is not a location file."
    return (ok >= min_rows, "" if ok >= min_rows else f"Only {ok} location lines found; expected one per mote.")

def fetch_locs(dest_dir, url=LOCS_URL, timeout=30):
    """Download and check mote_locs.txt into dest_dir. Returns (path or None, message). Nothing is left behind on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wsnfd"})
        with urllib.request.urlopen(req, timeout=timeout) as r: raw = r.read(1_000_000)
        text = raw.decode("utf-8", errors="strict")
    except Exception as e:
        return None, (f"Could not download it ({type(e).__name__}: {e}). Open {url} in your browser, press Ctrl+S, "
                      "save it as mote_locs.txt next to the data file, then press Browse.")
    ok, why = validate_locs(text)
    if not ok: return None, f"The downloaded file was not a valid location file. {why}"
    os.makedirs(dest_dir, exist_ok=True); out = os.path.join(dest_dir, "mote_locs.txt")
    fd, tmp = tempfile.mkstemp(dir=dest_dir); os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f: f.write(text)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return out, f"Downloaded mote_locs.txt ({sum(1 for l in text.splitlines() if l.strip())} motes) from the Intel Lab site."
