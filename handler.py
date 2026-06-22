"""RunPod Serverless handler for VoxCPM2. Starts the nano-vLLM server in-process and
proxies each job to it. concurrency_modifier lets one worker batch many cues at once
(exploiting nano-vLLM continuous batching, max_num_seqs)."""
import base64, subprocess, time, os
import requests
import runpod

SERVER = "http://127.0.0.1:8000/generate"

def _start_server():
    subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="/opt/nanovllm-voxcpm/deployment",
    )

def _wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            # zero-shot warmup generate confirms the model is loaded on GPU
            r = requests.post(SERVER, json={"target_text": "warmup", "cfg_value": 2.0}, timeout=20)
            if r.status_code == 200 and len(r.content) > 200:
                print(f"[handler] server ready after {time.time()-t0:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print("[handler] server NOT ready within timeout", flush=True)
    return False

_start_server()
_wait_ready()

def handler(job):
    inp = job.get("input", {})
    try:
        r = requests.post(SERVER, json=inp, timeout=300)
        if r.status_code == 200 and len(r.content) > 200:
            return {"audio_b64": base64.b64encode(r.content).decode(), "bytes": len(r.content)}
        return {"error": r.status_code, "body": r.text[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}

# allow one worker to process up to 24 jobs concurrently (nano-vLLM batches them)
runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda current: 24})
