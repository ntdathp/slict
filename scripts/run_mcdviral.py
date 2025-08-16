#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for MCDVIRAL sequences with low-lag logging.

Key optimizations:
- Remove `tee`: write roslaunch output directly to a log file with a large buffer.
- Reduce ROS console verbosity via ROSCONSOLE_CONFIG_FILE (WARN by default).
- Two options for /liteloam_pose:
  * RECORD_MODE (recommended): record to a bag during run, post-convert to CSV after.
  * DIRECT_CSV: stream CSV directly (simpler but a bit more I/O at runtime).
- Robust cleanup of child processes via process groups (os.setsid + killpg).
"""

import os
import subprocess
import time
import signal
import sys
from typing import List

# =====================
# Configuration
# =====================

# Root directory containing the dataset subfolders
DATASET_ROOT      = "/home/dat/Downloads/data/MCDVIRAL"

# Path to the ROS launch file (package XML/args inside should accept data_path & bag_file)
LAUNCH_FILE       = "/home/dat/slict_ws/src/slict/launch/run_mcdviral_uloc.launch"

# Delay between sequences (seconds)
DELAY_BETWEEN     = 5

# Timeout waiting for the first /lastcloud message (seconds)
LASTCLOUD_TIMEOUT = 20

# Choose how to capture /liteloam_pose:
# - True: record to bag during run, then export CSV after run (least runtime overhead)
# - False: write CSV live via `rostopic echo -p`
USE_RECORD_MODE   = True

# CSV & LOG buffering (in bytes). Large buffers reduce syscalls/IO thrash.
FILE_BUFFER_BYTES = 1024 * 1024  # 1 MB

# Reduce ROS console chatter (WARN or ERROR). Nodes still write full logs under ~/.ros/log.
ROS_CONSOLE_LEVEL = "WARN"

# =====================
# Globals
# =====================

# Track *all* child processes so we can terminate on Ctrl-C
child_procs: List[subprocess.Popen] = []

def kill_proc_group_safely(proc: subprocess.Popen, sig=signal.SIGTERM) -> None:
    """
    Kill the entire process group of 'proc' with the given signal.
    Any errors (already dead, etc.) are ignored.
    """
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except Exception:
        pass

def cleanup_all_and_exit(signum, frame):
    """
    Signal handler for SIGINT/SIGTERM:
    Terminate *all* known child processes and exit immediately.
    """
    print("\n[INFO] Signal received, terminating all child processes...")
    for p in child_procs:
        kill_proc_group_safely(p, signal.SIGTERM)
    sys.exit(0)

# Register global cleanup handlers
signal.signal(signal.SIGINT,  cleanup_all_and_exit)
signal.signal(signal.SIGTERM, cleanup_all_and_exit)

def ensure_rosconsole_config(level: str) -> str:
    """
    Create a temporary ROS console config to reduce console verbosity.
    Returns the path to the config file.
    """
    cfg_path = "/tmp/rosconsole.cfg"
    try:
        with open(cfg_path, "w") as f:
            # This line reduces verbosity for all ROS loggers on console
            f.write(f"log4j.logger.ros={level}\n")
    except Exception as e:
        print(f"[WARN] Failed to write ROS console config: {e}")
    return cfg_path

def list_sequence_folders(root: str) -> List[str]:
    """
    Return sorted list of subfolders that contain 'day' or 'night'.
    """
    if not os.path.isdir(root):
        return []
    subs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and ("day" in d or "night" in d)
    ]
    subs.sort()
    return subs

def run_one_sequence(folder_path: str, folder_name: str) -> None:
    """
    Run roslaunch on a single sequence folder:
      - Start CSV capture (DIRECT) or rosbag record (RECORD_MODE).
      - Launch roslaunch with LOG buffering and reduced console verbosity.
      - Wait for /lastcloud (timeout -> abort this sequence gracefully).
      - On normal finish, stop capture, and if RECORD_MODE then export CSV.

    All processes for this sequence are cleaned up before returning.
    """
    bag_pattern = os.path.join(folder_path, "*.bag")
    log_path    = os.path.join(folder_path, f"{folder_name}.log")
    csv_path    = os.path.join(folder_path, f"{folder_name}_liteloam_pose.csv")
    bag_out     = os.path.join(folder_path, "liteloam_pose.bag")  # used only in RECORD_MODE

    print(f"\n=== Processing folder: {folder_name} ===")
    print(f"Log file: {log_path}")
    print(f"CSV: {csv_path}")

    # Prepare ROS console config to reduce console spam
    os.environ["ROSCONSOLE_CONFIG_FILE"] = ensure_rosconsole_config(ROS_CONSOLE_LEVEL)

    # Open the log file with a large buffer; append mode
    log_f = open(log_path, "a", buffering=FILE_BUFFER_BYTES)

    # ---- 1) Start pose capture (either RECORD or DIRECT CSV) ----
    echo_proc = None
    record_proc = None

    if USE_RECORD_MODE:
        # rosbag record will run until we stop it; SIGINT is preferred for clean bag closure
        record_proc = subprocess.Popen(
            ["rosbag", "record", "-O", bag_out, "/liteloam_pose"],
            stdout=log_f,                    # record also writes some status lines -> send to log
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid             # put into its own process group
        )
        child_procs.append(record_proc)
    else:
        # Direct CSV via rostopic echo -p
        csv_f = open(csv_path, "w", buffering=FILE_BUFFER_BYTES)
        echo_proc = subprocess.Popen(
            ["rostopic", "echo", "-p", "/liteloam_pose"],
            stdout=csv_f,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        child_procs.append(echo_proc)

    # ---- 2) Launch ROS application, write all output to the log file (no tee) ----
    # Prefer list-args form (no shell) to avoid unnecessary shell overhead
    launch_proc = subprocess.Popen(
        ["roslaunch", LAUNCH_FILE,
         f"data_path:={DATASET_ROOT}", f"bag_file:={bag_pattern}", "autorun:=1"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    child_procs.append(launch_proc)

    # ---- 3) Wait for the first /lastcloud message with timeout ----
    monitor_proc = subprocess.Popen(
        ["rostopic", "echo", "-n1", "/lastcloud"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    child_procs.append(monitor_proc)

    got_lastcloud = False
    try:
        monitor_proc.wait(timeout=LASTCLOUD_TIMEOUT)
        got_lastcloud = True
        print(f"[OK] /lastcloud received within {LASTCLOUD_TIMEOUT}s.")
    except subprocess.TimeoutExpired:
        print(f"[WARN] No /lastcloud within {LASTCLOUD_TIMEOUT}s. Aborting this sequence...")

    # ---- 4) If timeout -> abort this sequence gracefully; else wait for normal finish ----
    try:
        if not got_lastcloud:
            # Terminate launch early
            kill_proc_group_safely(launch_proc, signal.SIGTERM)
            # Give it a moment to exit cleanly
            try:
                launch_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_proc_group_safely(launch_proc, signal.SIGKILL)
        else:
            # Wait for roslaunch to complete normally
            launch_proc.wait()
    finally:
        # Stop capture processes (record or echo)
        if USE_RECORD_MODE and record_proc is not None:
            # SIGINT lets rosbag record finalize the bag file properly
            kill_proc_group_safely(record_proc, signal.SIGINT)
            try:
                record_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_proc_group_safely(record_proc, signal.SIGKILL)

            # Post-process: export CSV from the recorded bag
            try:
                # rostopic echo -b <bag> -p /liteloam_pose > csv_path
                with open(csv_path, "w", buffering=FILE_BUFFER_BYTES) as csv_out:
                    export_proc = subprocess.Popen(
                        ["rostopic", "echo", "-b", bag_out, "-p", "/liteloam_pose"],
                        stdout=csv_out,
                        stderr=subprocess.DEVNULL
                    )
                    export_proc.wait()
                print(f"[OK] Exported CSV from {bag_out} -> {csv_path}")
            except Exception as e:
                print(f"[ERROR] Failed to export CSV from bag: {e}")

        if not USE_RECORD_MODE and echo_proc is not None:
            kill_proc_group_safely(echo_proc, signal.SIGTERM)
            try:
                echo_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_proc_group_safely(echo_proc, signal.SIGKILL)

        # Stop the /lastcloud monitor if still alive
        kill_proc_group_safely(monitor_proc, signal.SIGTERM)
        try:
            monitor_proc.wait(timeout=3)
        except Exception:
            pass

        # Close log file to flush buffers
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass

    print(f"Completed: {folder_name}")

def main():
    """
    Main loop:
      - Verify ROS master.
      - Find all 'day'/'night' subfolders.
      - Run each sequence with low-lag settings.
    """
    if not os.environ.get("ROS_MASTER_URI"):
        print("ERROR: ROS_MASTER_URI is not set. Please start roscore first.")
        return

    subfolders = list_sequence_folders(DATASET_ROOT)
    if not subfolders:
        print("No 'day' or 'night' subfolders found. Exiting.")
        return

    print(f"Found {len(subfolders)} sequences.")
    print(f"Capture mode: {'RECORD_MODE (bag->CSV after run)' if USE_RECORD_MODE else 'DIRECT_CSV (live CSV)'}")

    for folder in subfolders:
        folder_path = os.path.join(DATASET_ROOT, folder)

        # Run one sequence, with robust per-sequence cleanup
        try:
            run_one_sequence(folder_path, folder)
        except KeyboardInterrupt:
            # Let the global handler take care of cleanup
            raise
        except Exception as e:
            print(f"[ERROR] Sequence '{folder}' failed: {e}")

        # Clear known child processes list for the next iteration
        child_procs.clear()

        # Small delay to avoid thrashing between runs
        time.sleep(DELAY_BETWEEN)

    print("\nAll folders processed. Exiting.")

if __name__ == "__main__":
    main()
