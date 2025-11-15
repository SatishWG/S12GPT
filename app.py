import gradio as gr
import threading
import subprocess
import sys
import time
import os
from typing import List

# simple background process manager to run train.py and capture stdout/stderr
_PROC = None
_PROC_THREAD = None
_LOGS: List[str] = []
_LOCK = threading.Lock()

TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
PY = sys.executable  # use same Python interpreter


def _reader_thread(proc: subprocess.Popen):
    global _LOGS
    try:
        for line in proc.stdout:
            if not line:
                continue
            with _LOCK:
                _LOGS.append(line.rstrip("\n"))
    except Exception as e:
        with _LOCK:
            _LOGS.append(f"[reader error] {e}")
    finally:
        proc.stdout.close()
        ret = proc.wait()
        with _LOCK:
            _LOGS.append(f"[process exited with code {ret}]")


def start_training(restart: bool, epochs: int):
    """
    Start train.py as a background process. Returns immediate status string.
    Use Refresh log button to see latest logs.
    """
    global _PROC, _PROC_THREAD, _LOGS

    if _PROC is not None and _PROC.poll() is None:
        return "Training already running."

    # build command; train.py currently reads internal config, so we just run it.
    # If you later modify train.py to accept args, update this command accordingly.
    cmd = [PY, TRAIN_SCRIPT]

    # clear logs
    with _LOCK:
        _LOGS = [f"[starting training: restart={restart}, epochs={epochs}]"]

    try:
        _PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        with _LOCK:
            _LOGS.append(f"[failed to start process] {e}")
        return f"Failed to start training: {e}"

    _PROC_THREAD = threading.Thread(target=_reader_thread, args=(_PROC,), daemon=True)
    _PROC_THREAD.start()
    return "Training started."


def stop_training():
    global _PROC, _PROC_THREAD
    if _PROC is None:
        return "No training process."
    if _PROC.poll() is not None:
        return f"Process already exited (code {_PROC.returncode})."
    try:
        _PROC.terminate()
        # give it a short time to exit gracefully
        try:
            _PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _PROC.kill()
        return "Termination signal sent."
    except Exception as e:
        return f"Failed to terminate: {e}"


def refresh_logs(max_lines: int = 1000):
    with _LOCK:
        if not _LOGS:
            return ""
        # return last max_lines lines
        lines = _LOGS[-max_lines:]
        return "\n".join(lines)


with gr.Blocks() as app:
    gr.Markdown("# Train GPT (train.py)\nRun training and stream logs (press Refresh to poll latest logs).")
    with gr.Row():
        restart_chk = gr.Checkbox(label="Restart (ignore any checkpoints)", value=False)
        epochs_num = gr.Number(label="Epochs (informational)", value=100, precision=0)
    with gr.Row():
        start_btn = gr.Button("Start training")
        stop_btn = gr.Button("Stop training")
        refresh_btn = gr.Button("Refresh logs")
    logs_box = gr.Textbox(label="Training log", lines=30, interactive=False)

    start_btn.click(fn=start_training, inputs=[restart_chk, epochs_num], outputs=logs_box)
    stop_btn.click(fn=stop_training, inputs=[], outputs=logs_box)
    refresh_btn.click(fn=refresh_logs, inputs=[], outputs=logs_box)

# expose for Hugging Face Spaces
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)