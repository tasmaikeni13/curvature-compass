"""Cache primary papers locally; preserve hashes, never redistribute paper text."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

def fetch(source):
    path = CACHE / (source["id"] + ".pdf")
    result = dict(source)
    try:
        if not path.exists():
            request = urllib.request.Request(source["url"], headers={"User-Agent": "CurvatureCompass-Phase1/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                result["resolved_url"] = response.url
            if not data.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            path.write_bytes(data)
        data = path.read_bytes()
        result["sha256"] = hashlib.sha256(data).hexdigest()
        result["bytes"] = len(data)
        subprocess.run(["pdftotext", "-layout", str(path), str(path.with_suffix(".txt"))], check=True, capture_output=True)
        result["text_lines"] = len(path.with_suffix(".txt").read_text().splitlines())
        result["status"] = "downloaded_and_extracted"
    except Exception as error:
        result["status"] = "unavailable"
        result["error"] = str(error)
    return result

if __name__ == "__main__":
    sources = json.loads((ROOT / "sources.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, sources))
    manifest = {"retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "records": records}
    (ROOT / "retrieval.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for record in records:
        print(record["id"], record["status"], record.get("text_lines", record.get("error")))
