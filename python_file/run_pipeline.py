#!/usr/bin/env python3
import subprocess
import sys
import re

def run_command(command, capture_output=False):
    """Runs a shell command, streams output, and optionally captures it."""
    print(f"\n---> Running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if capture_output else sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 # Line buffered
    )

    stdout_lines = []
    stderr_lines = []

    # Stream stderr in real-time
    stderr_thread = None
    if process.stderr:
        def stream_stderr():
            for line in process.stderr:
                print(line, end='', file=sys.stderr)
                stderr_lines.append(line)
        import threading
        stderr_thread = threading.Thread(target=stream_stderr)
        stderr_thread.start()

    # Process stdout (stream or capture)
    if capture_output and process.stdout:
        for line in process.stdout:
            print(line, end='') # Still print it for visibility
            stdout_lines.append(line)
    elif not capture_output and process.stdout:
         # If not capturing, just let Popen stream to sys.stdout directly
         pass # stdout is already going to sys.stdout

    # Wait for process to finish and stderr thread
    return_code = process.wait()
    if stderr_thread:
        stderr_thread.join()

    print(f"---> Command finished with exit code: {return_code}")
    if return_code != 0:
        print("Error during command execution. Aborting pipeline.", file=sys.stderr)
        sys.exit(return_code)

    return "".join(stdout_lines) if capture_output else None

def main():
    # --- Configuration ---
    train_script = "train.py"
    predict_script = "predict_hvac.py"

    # Pass any arguments from this script to train.py
    train_args = sys.argv[1:]

    # --- Run Training ---
    train_command = [sys.executable, train_script] + train_args
    # Capture output to get the Run ID
    train_output = run_command(train_command, capture_output=True)

    # --- Extract Run ID ---
    run_id = None
    if train_output:
        match = re.search(r"MLFLOW_RUN_ID:(\S+)", train_output)
        if match:
            run_id = match.group(1)
            print(f"\n---> Extracted Training Run ID: {run_id}")
        else:
            print("Error: Could not find MLFLOW_RUN_ID in train.py output.", file=sys.stderr)
            sys.exit(1)
    else:
         print("Error: No output captured from train.py.", file=sys.stderr)
         sys.exit(1)


    # --- Run Prediction ---
    predict_command = [
        sys.executable,
        predict_script,
        "--training_run_id",
        run_id, 
        "--output_filename",
        f"quan_submission_{run_id}.csv"
        # Add other predict.py specific arguments here if needed
        # e.g., "--output_filename", "custom_submission.csv"
    ]
    print(f"\n---> Running: {' '.join(predict_command)}")
    run_command(predict_command)

    print("\n---> Pipeline finished successfully!")

if __name__ == "__main__":
    main() 