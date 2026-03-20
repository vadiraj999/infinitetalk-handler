"""
handler.py — RunPod Serverless Handler for InfiniteTalk
========================================================
Accepts: base64-encoded image + base64-encoded audio (mp3/wav)
Returns: base64-encoded mp4 video

RunPod calls runpod.serverless.start({"handler": handler})
Input JSON:
{
    "input": {
        "image": "<base64 string>",
        "image_ext": "jpg",          // optional, default "jpg"
        "audio": "<base64 string>",
        "audio_ext": "mp3",          // optional, default "mp3"
        "resolution": "480p",        // optional, default "480p"
        "sample_steps": 20,          // optional, default 20 (faster) or 40 (higher quality)
        "motion_frame": 9            // optional, default 9
    }
}
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import runpod

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR     = Path(os.environ.get("WEIGHTS_DIR", "/workspace/weights"))
INFINITETALK_DIR = Path("/infinitetalk")   # cloned repo baked into Docker image

CKPT_DIR      = WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"
WAV2VEC_DIR   = WEIGHTS_DIR / "chinese-wav2vec2-base"
IT_WEIGHTS    = WEIGHTS_DIR / "InfiniteTalk" / "single" / "infinitetalk.safetensors"

# ── Validate weights on cold start ────────────────────────────────────────────
def validate_weights():
    missing = []
    for p in [CKPT_DIR, WAV2VEC_DIR, IT_WEIGHTS]:
        if not Path(p).exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError(
            f"Missing model weights — did you run setup_weights.sh?\n"
            + "\n".join(f"  ✗ {m}" for m in missing)
        )
    print("[handler] ✓ All weights found", flush=True)


# ── Main handler ──────────────────────────────────────────────────────────────
def handler(job):
    validate_weights()  # check weights on every job — safe, fast, catches mount issues early
    job_input = job.get("input", {})
    t_start   = time.time()

    # ── Decode inputs ─────────────────────────────────────────────────────────
    image_b64  = job_input.get("image")
    audio_b64  = job_input.get("audio")
    image_ext  = job_input.get("image_ext", "jpg").lstrip(".")
    audio_ext  = job_input.get("audio_ext", "mp3").lstrip(".")
    resolution = job_input.get("resolution", "480p")
    steps      = int(job_input.get("sample_steps", 20))
    motion_f   = int(job_input.get("motion_frame", 9))

    if not image_b64:
        return {"error": "Missing 'image' in input"}
    if not audio_b64:
        return {"error": "Missing 'audio' in input"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write image and audio to disk
        image_path  = tmpdir / f"input_image.{image_ext}"
        audio_path  = tmpdir / f"input_audio.{audio_ext}"
        output_path = tmpdir / "output.mp4"
        input_json  = tmpdir / "input.json"

        image_path.write_bytes(base64.b64decode(image_b64))
        audio_path.write_bytes(base64.b64decode(audio_b64))

        print(f"[handler] Image: {image_path.name} ({image_path.stat().st_size/1024:.0f} KB)", flush=True)
        print(f"[handler] Audio: {audio_path.name} ({audio_path.stat().st_size/1024:.0f} KB)", flush=True)

        # Build the input JSON that InfiniteTalk expects
        input_data = [
            {
                "image": str(image_path),
                "audio": str(audio_path)
            }
        ]
        input_json.write_text(json.dumps(input_data))

        # Build the generate command
        cmd = [
            sys.executable,
            str(INFINITETALK_DIR / "generate_infinitetalk.py"),
            "--ckpt_dir",          str(CKPT_DIR),
            "--wav2vec_dir",       str(WAV2VEC_DIR),
            "--infinitetalk_dir",  str(IT_WEIGHTS),
            "--input_json",        str(input_json),
            "--size",              f"infinitetalk-{resolution.replace('p','')}",
            "--sample_steps",      str(steps),
            "--num_persistent_param_in_dit", "0",
            "--mode",              "streaming",
            "--motion_frame",      str(motion_f),
            "--save_file",         str(tmpdir / "output"),
        ]

        print(f"[handler] Running InfiniteTalk (steps={steps}, res={resolution})...", flush=True)
        t_gen = time.time()

        # Pass env vars explicitly so HF_HOME reaches the subprocess
        import copy
        sub_env = copy.deepcopy(dict(os.environ))
        hf_cache = str(Path(os.environ.get("WEIGHTS_DIR", "/runpod-volume/weights")).parent / "hf_cache")
        sub_env["HF_HOME"] = hf_cache
        sub_env["TRANSFORMERS_CACHE"] = hf_cache
        sub_env["HF_HUB_OFFLINE"] = "1"
        sub_env["TRANSFORMERS_OFFLINE"] = "1"
        sub_env["WEIGHTS_DIR"] = str(WEIGHTS_DIR)

        result = subprocess.run(
            cmd,
            cwd=str(INFINITETALK_DIR),
            capture_output=False,
            timeout=7200,
            env=sub_env,
        )

        if result.returncode != 0:
            return {"error": f"InfiniteTalk exited with code {result.returncode}"}

        # InfiniteTalk appends _0.mp4 to the save_file path
        candidates = list(tmpdir.glob("output*.mp4"))
        if not candidates:
            return {"error": "No output video found after generation"}

        output_path = candidates[0]
        gen_time    = time.time() - t_gen
        size_mb     = output_path.stat().st_size / 1024 / 1024

        print(f"[handler] ✓ Generated in {gen_time:.0f}s — {size_mb:.1f} MB", flush=True)

        # Encode output to base64
        video_b64 = base64.b64encode(output_path.read_bytes()).decode("utf-8")

        total_time = time.time() - t_start
        return {
            "video": video_b64,
            "video_ext": "mp4",
            "generation_time_seconds": round(gen_time),
            "total_time_seconds": round(total_time),
            "size_mb": round(size_mb, 1),
        }


# ── Entry point ───────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
