#!/usr/bin/env python3
"""Environment and dependency verification script for RobustRAG."""

import os
import platform
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustrag.utils import load_env_variables


def check_environment() -> bool:
    print("=" * 60)
    print(" RobustRAG System Environment Verification")
    print("=" * 60)

    # 1. Python version check
    py_version = sys.version_info
    print(f"[*] Python Version: {py_version.major}.{py_version.minor}.{py_version.micro} ({platform.platform()})")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 11):
        print("  [!] WARNING: Recommended Python version is >= 3.11")
    else:
        print("  [+] Python version verified.")

    # 2. Package imports
    core_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("sentence_transformers", "Sentence Transformers (Cross-Encoder)"),
        ("rank_bm25", "Rank-BM25 (Lexical Retrieval)"),
        ("sklearn", "Scikit-learn"),
        ("scipy", "SciPy (Statistical Bootstrapping)"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("streamlit", "Streamlit"),
        ("yaml", "PyYAML"),
        ("groq", "Groq Client (Fast LLM backend)"),
    ]

    all_passed = True
    print("\n[*] Checking Core Libraries:")
    for mod_name, desc in core_packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [+] {desc} (version: {ver})")
        except ImportError as e:
            print(f"  [-] MISSING: {desc} ({e})")
            all_passed = False

    # 3. Acceleration device
    print("\n[*] Checking Hardware Device:")
    try:
        import torch
        if torch.cuda.is_available():
            dev = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            dev = "Apple Silicon MPS (Metal Performance Shaders)"
        else:
            dev = "Standard CPU"
        print(f"  [+] Detected Acceleration Device: {dev}")
    except Exception as e:
        print(f"  [!] PyTorch device check skipped: {e}")

    # 4. Storage permissions
    print("\n[*] Checking Write Permissions:")
    out_dir = Path("outputs")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".perm_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print("  [+] Output directory permissions verified.")
    except Exception as e:
        print(f"  [-] Output directory write error: {e}")
        all_passed = False

    # 5. Dataset Fallback Check
    fallback_file = Path("data/squad_sample_fallback.json")
    if fallback_file.exists():
        print(f"  [+] Offline fallback dataset verified ({fallback_file}).")
    else:
        print(f"  [!] Fallback dataset missing at {fallback_file}.")

    # 6. Groq Key Check
    load_env_variables()
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        masked = groq_key[:8] + "..." + groq_key[-4:]
        print(f"  [+] GROQ_API_KEY detected ({masked}) - Groq acceleration enabled.")
    else:
        print("  [*] GROQ_API_KEY not configured - Local Flan-T5 backend will be used.")

    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL CHECKS PASSED: Environment is ready for RobustRAG research!")
    else:
        print(" [!] SOME CHECKS FAILED: Please install missing dependencies.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
