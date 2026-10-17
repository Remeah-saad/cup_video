"""
check_dependencies.py

Checks whether the packages this project needs are already importable in the
ACTIVE Python environment. Only missing packages are installed - nothing is
reinstalled or upgraded if it's already present, and no assumption is made
about the operating system (uses `sys.executable -m pip`, which works the
same on Windows/macOS/Linux).

Run this before anything else:
    python check_dependencies.py
"""

import importlib
import subprocess
import sys

# (import_name, pip_install_name) -- these can differ (e.g. cv2 vs opencv-contrib-python)
REQUIRED = [
    ("cv2", "opencv-contrib-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("streamlit", "streamlit"),
    ("matplotlib", "matplotlib"),
]


def main():
    missing = []
    present = []

    for import_name, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            present.append((import_name, version))
        except ImportError:
            missing.append(pip_name)

    print("Already installed:")
    for name, version in present:
        print(f"  [OK] {name} ({version})")

    if not missing:
        print("\nAll dependencies are already installed. Nothing to do.")
        return

    print("\nMissing packages:")
    for pkg in missing:
        print(f"  [MISSING] {pkg}")

    answer = input(
        f"\nInstall the {len(missing)} missing package(s) now with "
        f"'{sys.executable} -m pip install'? [y/N]: "
    ).strip().lower()

    if answer != "y":
        print("Skipped installation. Install manually when ready:")
        print(f"  {sys.executable} -m pip install " + " ".join(missing))
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print("\nDone. Re-run this script to verify.")


if __name__ == "__main__":
    main()
