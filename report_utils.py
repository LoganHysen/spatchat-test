# report_utils.py

from typing import List, Tuple, Optional
import os
import io
import shutil
import zipfile
import numpy as np

outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

def add_to_report(parts: List[str], title: str, html_fragment: str):
    parts.append(f"<h2>{title}</h2>{html_fragment}")

def write_report(sections: List[Tuple[str, str, Optional[np.ndarray]]]) -> str:
    """
    sections: list of (title, preformatted_text, optional_image_np)
    Writes an HTML report to outputs/stats_report.html and returns its filepath.
    """
    parts = ["<h1>SpatChat – Stats Report</h1>"]
    for title, html_text, img_np in sections:
        add_to_report(parts, title, f"<pre style='white-space:pre-wrap'>{html_text}</pre>")
        if img_np is not None:
            import base64
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(img_np).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto' />")
    html = "\n".join(parts).encode("utf-8")
    fp = os.path.join(outputs_dir, "stats_report.html")
    with open(fp, "wb") as f:
        f.write(html)
    return fp

def save_image_np(arr: np.ndarray, fname: str) -> str:
    """
    Save a numpy RGB image array to outputs/<fname> (PNG) and return the filepath.
    """
    from PIL import Image
    fp = os.path.join(outputs_dir, fname)
    Image.fromarray(arr).save(fp, format="PNG")
    return fp

def save_zip() -> str:
    """
    Zips everything currently in outputs/ (excluding the zip itself) to
    outputs/spatchat_stats_results.zip and returns the archive path.
    """
    archive = os.path.join(outputs_dir, "spatchat_stats_results.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(outputs_dir):
            for fn in files:
                if fn.endswith(".zip"):
                    continue
                full = os.path.join(root, fn)
                z.write(full, arcname=fn)
    return archive

def clear_outputs():
    """
    Clears and recreates outputs/ directory.
    """
    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)
