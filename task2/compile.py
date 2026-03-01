import os
import shutil
import subprocess
import sys


def compile():
    setup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda"))
    if shutil.which("uv") is not None:
        cmd = ["uv", "pip", "install", "-e", ".", "--no-build-isolation"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"]
    print(f"Running in {setup_path}: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=setup_path)


if __name__ == "__main__":
    compile()
