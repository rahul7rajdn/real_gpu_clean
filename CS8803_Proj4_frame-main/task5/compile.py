import os
import shutil
import subprocess
import sys


def _install_extension(cuda_dir):
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "-e", ".", "--no-build-isolation"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"]

    print(f"Running in {cuda_dir}: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cuda_dir)


def compile():
    task_root = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(task_root, ".."))

    # Task 5 test imports both prefill (task4) and decode (task5) extensions.
    task4_cuda = os.path.join(project_root, "task4", "cuda")
    task5_cuda = os.path.join(project_root, "task5", "cuda")

    _install_extension(task4_cuda)
    _install_extension(task5_cuda)


if __name__ == "__main__":
    compile()
