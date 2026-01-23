from pathlib import Path
from typing import Optional
from urllib.parse import quote
from uuid import uuid4
import mimetypes
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from src.config import IMAGE_DIR, TOP_K, UPLOAD_DIR
from src.pipeline import DuplicateDetector, build_hash_db

app = FastAPI(title="Near Duplicate Image UI", version="0.1.0")


def _resolve_safe_path(raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    allowed_roots = [Path(IMAGE_DIR).resolve(), Path(UPLOAD_DIR).resolve()]

    if not any(str(path).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Path not allowed")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return path


@app.on_event("startup")
def init_detector() -> None:
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    hash_db = build_hash_db(IMAGE_DIR)
    app.state.detector = DuplicateDetector(image_dir=IMAGE_DIR, hash_db=hash_db)


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload an image file")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    suffix = Path(file.filename).suffix or ".jpg"
    temp_path = Path(UPLOAD_DIR) / f"{uuid4().hex}{suffix}"

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detector: DuplicateDetector = app.state.detector
    result = detector.detect(str(temp_path), top_k=TOP_K)

    def with_preview(path: Optional[str] = None) -> Optional[str]:
        return f"/preview?path={quote(path)}" if path else None

    result["uploaded_url"] = with_preview(str(temp_path))
    result["match_url"] = with_preview(result.get("match"))

    for item in result.get("verifier_matches", []):
        item["preview_url"] = with_preview(item.get("filename"))

    for item in result.get("sieve_matches", []):
        item["preview_url"] = with_preview(item.get("filename"))

    return result


@app.get("/preview")
def preview_image(path: str) -> FileResponse:
    file_path = _resolve_safe_path(path)
    media_type, _ = mimetypes.guess_type(file_path.name)
    return FileResponse(file_path, media_type=media_type or "image/jpeg")


@app.get("/")
def index() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Near Duplicate Finder</title>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap\" rel=\"stylesheet\" />
        <style>
            :root {
                --bg: #0d0f1a;
                --card: rgba(20, 24, 40, 0.85);
                --accent: #6ee7ff;
                --accent-2: #c084fc;
                --muted: #9ba3b4;
                --text: #e9ecf5;
                --border: rgba(255, 255, 255, 0.08);
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                min-height: 100vh;
                font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
                color: var(--text);
                background: radial-gradient(circle at 10% 20%, rgba(110, 231, 255, 0.12), transparent 30%),
                            radial-gradient(circle at 90% 10%, rgba(192, 132, 252, 0.1), transparent 28%),
                            linear-gradient(135deg, #0c0f1e 0%, #0b1226 55%, #0a0c1a 100%);
                padding: 24px;
            }
            h1 {
                margin: 0 0 8px 0;
                font-size: 28px;
                letter-spacing: -0.01em;
            }
            p.lede {
                margin: 0 0 24px 0;
                color: var(--muted);
                max-width: 680px;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1.1fr;
                gap: 20px;
            }
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 14px 40px rgba(0,0,0,0.35);
            }
            .upload-card {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .dropzone {
                border: 1.5px dashed var(--border);
                border-radius: 14px;
                padding: 28px;
                text-align: center;
                cursor: pointer;
                transition: border-color 0.2s ease, background 0.2s ease;
            }
            .dropzone:hover {
                border-color: var(--accent);
                background: rgba(110, 231, 255, 0.06);
            }
            .button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: #0b0d18;
                border: none;
                border-radius: 12px;
                padding: 12px 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.15s ease, box-shadow 0.2s ease;
            }
            .button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 30px rgba(110, 231, 255, 0.25);
            }
            .preview {
                width: 100%;
                border-radius: 12px;
                background: #0a0c16;
                border: 1px solid var(--border);
                aspect-ratio: 4 / 3;
                object-fit: contain;
            }
            .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: rgba(255,255,255,0.02);
            }
            .status.positive { color: #8ef5a2; border-color: rgba(142,245,162,0.4); }
            .status.negative { color: #f59e0b; border-color: rgba(245,158,11,0.4); }
            .matches {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin-top: 12px;
            }
            .match-card {
                background: rgba(255,255,255,0.03);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 10px;
            }
            .match-card img {
                width: 100%;
                height: 160px;
                object-fit: cover;
                border-radius: 10px;
                border: 1px solid var(--border);
            }
            .muted { color: var(--muted); }
            .path { color: var(--muted); font-size: 12px; margin-top: 6px; }
            .score { font-weight: 600; margin-bottom: 6px; }
            @media (max-width: 900px) {
                .grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <h1>Near Duplicate Finder</h1>
        <p class=\"lede\">Upload an image to run the two-stage sieve (dHash) and verifier (SSCD + FAISS). The UI will show the top candidates and flag duplicates when thresholds are met.</p>
        <div class=\"grid\">
            <div class=\"card upload-card\">
                <div class=\"dropzone\" id=\"dropzone\">
                    <p><strong>Drop an image</strong> or click to browse.</p>
                    <p class=\"muted\">JPEG, PNG, WEBP</p>
                    <input type=\"file\" id=\"file-input\" accept=\"image/*\" style=\"display:none\" />
                </div>
                <button class=\"button\" id=\"submit-btn\">Run similarity search</button>
                <div>
                    <p class=\"muted\" style=\"margin: 0 0 6px 0;\">Query preview</p>
                    <img id=\"query-preview\" class=\"preview\" alt=\"Query preview\" />
                </div>
            </div>
            <div class=\"card\">
                <div id=\"status\" class=\"status\">Waiting for an upload.</div>
                <div id=\"best-match\" style=\"margin-top: 14px; display: none;\">
                    <p class=\"muted\" style=\"margin: 0 0 6px 0;\">Best candidate</p>
                    <img id=\"best-preview\" class=\"preview\" alt=\"Best match\" />
                    <div id=\"best-meta\" class=\"path\"></div>
                </div>
                <div style=\"margin-top: 16px;\">
                    <p class=\"muted\" style=\"margin: 0 0 6px 0;\">Top results</p>
                    <div id=\"matches\" class=\"matches\"></div>
                </div>
            </div>
        </div>

        <script>
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('file-input');
            const submitBtn = document.getElementById('submit-btn');
            const statusEl = document.getElementById('status');
            const previewEl = document.getElementById('query-preview');
            const bestWrap = document.getElementById('best-match');
            const bestImg = document.getElementById('best-preview');
            const bestMeta = document.getElementById('best-meta');
            const matchesEl = document.getElementById('matches');

            dropzone.addEventListener('click', () => fileInput.click());
            dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--accent)'; });
            dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = 'var(--border)'; });
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.style.borderColor = 'var(--border)';
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    setPreview(fileInput.files[0]);
                }
            });

            fileInput.addEventListener('change', () => {
                if (fileInput.files.length) {
                    setPreview(fileInput.files[0]);
                }
            });

            submitBtn.addEventListener('click', async () => {
                if (!fileInput.files.length) {
                    statusEl.textContent = 'Pick an image to start.';
                    return;
                }

                const file = fileInput.files[0];
                statusEl.className = 'status';
                statusEl.textContent = 'Running search...';
                bestWrap.style.display = 'none';
                matchesEl.innerHTML = '';

                const form = new FormData();
                form.append('file', file);

                try {
                    const res = await fetch('/api/detect', { method: 'POST', body: form });
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Request failed');
                    }
                    const data = await res.json();
                    renderResults(data);
                } catch (err) {
                    statusEl.className = 'status negative';
                    statusEl.textContent = err.message || 'Something went wrong';
                }
            });

            function setPreview(file) {
                const url = URL.createObjectURL(file);
                previewEl.src = url;
            }

            function renderResults(data) {
                const duplicate = data.is_duplicate;
                const stage = data.stage;
                const score = data.score;
                statusEl.className = duplicate ? 'status positive' : 'status negative';
                statusEl.textContent = duplicate ? `Duplicate flagged at ${stage} stage` : 'No duplicate detected.';

                if (data.match_url) {
                    bestWrap.style.display = 'block';
                    bestImg.src = data.match_url;
                    const label = stage === 'sieve' ? `Hamming distance: ${data.score}` : `Similarity: ${score?.toFixed(3)}`;
                    bestMeta.textContent = `${label} · ${data.match}`;
                }

                const verifier = data.verifier_matches || [];
                if (verifier.length === 0) {
                    matchesEl.innerHTML = '<p class="muted">No index results yet. Build the FAISS index first.</p>';
                    return;
                }

                matchesEl.innerHTML = verifier.map((item, idx) => {
                    const scoreLabel = typeof item.score === 'number' ? item.score.toFixed(3) : item.distance;
                    const pathLabel = item.filename || 'Unknown file';
                    return `
                        <div class="match-card">
                            <div class="score">#${idx + 1} · ${scoreLabel}</div>
                            <img src="${item.preview_url}" alt="match ${idx + 1}" />
                            <div class="path">${pathLabel}</div>
                        </div>
                    `;
                }).join('');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
