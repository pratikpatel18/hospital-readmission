"""ONE command from a fresh clone:  python run_all.py
installs deps -> generates data -> trains/tunes/explains -> runs unit tests -> prints how to start API + UI.
Flags: --skip-install  --fast (small, ~1 min)  --serve (start API + Streamlit afterwards)"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--skip-install", action="store_true")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--serve", action="store_true")
a = ap.parse_args()


def sh(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if not a.skip_install:
    sh(sys.executable, "-m", "pip", "install", "-r", "requirements.txt")
sh(sys.executable, "-m", "src.train", *(["--fast", "--n-admissions", "20000"] if a.fast else []))
sh(sys.executable, "-m", "pytest", "-q")
print("\nAll done. Start the services with:\n  uvicorn api.main:app --port 8000\n  streamlit run app/streamlit_app.py")
if a.serve:
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"])
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    finally:
        api.terminate()
