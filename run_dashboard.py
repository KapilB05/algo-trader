import sys
import uvicorn
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"\n[+] Launching Quant Mission Control Terminal on http://localhost:{port} (or http://127.0.0.1:{port})...")
    uvicorn.run(
        "src.dashboard_v2.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,
    )

if __name__ == "__main__":
    main()

