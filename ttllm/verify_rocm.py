#!/usr/bin/env python
"""Check the ROCm 10 audio runtime without downloading or loading models."""

import os
from pathlib import Path
import sys


def main() -> int:
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    # Follow the active alternatives link, not a possibly inactive core-10.0.
    version_file = next(
        (p for p in (root / "core/.info/version", root / ".info/version")
         if p.is_file()), None
    )
    if version_file is None:
        print(f"FAIL: ROCm product version not found under {root}", file=sys.stderr)
        return 1
    version = version_file.read_text().strip()
    print(f"System ROCm: {version} ({version_file.resolve()})", flush=True)
    if version.split(".")[0] != "10":
        print("FAIL: active system ROCm is not 10.x", file=sys.stderr)
        return 1
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        print("FAIL: unset HSA_OVERRIDE_GFX_VERSION for native gfx1151", file=sys.stderr)
        return 1

    # Ordering is essential: CT2 must reuse torch's bundled runtime.
    import torch
    import torchaudio
    import ctranslate2

    print(f"torch: {torch.__version__}; bundled HIP: {torch.version.hip}")
    print(f"torchaudio: {torchaudio.__version__}; CT2: {ctranslate2.__version__}")
    if not all(hasattr(torchaudio, name) for name in ("info", "AudioMetaData")):
        raise RuntimeError("torchaudio lacks APIs required by pyannote; keep 2.8.x")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm GPU unavailable")
    x = torch.ones((32, 32), device="cuda")
    if not torch.allclose(x @ x, torch.full_like(x, 32)):
        raise RuntimeError("GPU matrix multiplication failed")
    count = ctranslate2.get_cuda_device_count()
    if count < 1:
        raise RuntimeError("CTranslate2 GPU unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}; CT2 devices: {count}")
    print("Loaded HIP/ROCm libraries (/proc/self/maps):")
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
    for path in sorted(paths):
        if path.startswith("/") and ".so" in path and any(
            word in path.lower() for word in ("hip", "rocm", "rocblas", "hsa", "amd")
        ):
            print(f"  {path}")
    print("OK: ROCm 10 runtime smoke check; run verify_coexist.py for ASR inference")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
