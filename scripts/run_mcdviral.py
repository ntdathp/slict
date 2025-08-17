#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for MCDVIRAL sequences (with tee):
- Terminal output is visible AND saved to a single file per sequence: exp_log_dir/<seq>/sequence.log
- No CSV export and no extra .log from this script beyond sequence.log
- Safe process-group cleanup; /lastcloud monitor with timeout
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

DATASET_ROOT      = "/home/dat/Downloads/data/MCDVIRAL"
LAUNCH_FILE       = "/home/dat/slict_ws/src/slict/launch/run_mcdviral_uloc.launch"
DELAY_BETWEEN     = 5
LASTCLOUD_TIMEOUT = 20
ROS_CONSOLE_LEVEL = "WARN"

# Base dir for per-sequence logs (align with your launch arg structure)
USER_NAME         = os.environ.get("USER", "dat")
EXP_LOG_BASE      = f"/home/{USER_NAME}/slict_logs/mcdviral"

# If you truly want ONLY one log file (sequence.log),
# keep this True and make sure nodes in the .launch use output="screen".
ONLY_ONE_SEQ_LOG  = True

# =====================
# Globals
# =====================

child_procs: List[subprocess.Popen] = []

def kill_proc_group_safely(proc: subprocess.Popen, sig=signal.SIGTERM) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except Exception:
        pass

def cleanup_all_and_exit(signum, frame):
    print("\n[INFO] Signal received, terminating all child processes...")
    for p in child_procs:
        kill_proc_group_safely(p, signal.SIGTERM)
    sys.exit(0)

signal.signal(signal.SIGINT,  cleanup_all_and_exit)
signal.signal(signal.SIGTERM, cleanup_all_and_exit)

def ensure_rosconsole_config(level: str) -> str:
    cfg_path = "/tmp/rosconsole.cfg"
    try:
        with open(cfg_path, "w") as f:
            f.write(f"log4j.logger.ros={level}\n")
    except Exception as e:
        print(f"[WARN] Failed to write ROS console config: {e}")
    return cfg_path

def list_sequence_folders(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    subs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and ("day" in d or "night" in d)
    ]
    subs.sort()
    return subs

def run_one_sequence(folder_path: str, folder_name: str) -> None:
    # rosbag pattern for this folder
    bag_pattern = os.path.join(folder_path, "*.bag")

    # exp_log_dir per sequence (match your launch arg)
    exp_log_dir = os.path.join(EXP_LOG_BASE, folder_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    log_path = os.path.join(exp_log_dir, "sequence.log")

    print(f"\n=== Processing folder: {folder_name} ===")
    print(f"exp_log_dir: {exp_log_dir}")
    print(f"sequence log: {log_path}")

    # Reduce console verbosity but still print to terminal
    os.environ["ROSCONSOLE_CONFIG_FILE"] = ensure_rosconsole_config(ROS_CONSOLE_LEVEL)

    # If you set ROS_LOG_DIR, nodes with output="log" will write there (extra files).
    # To keep *only* sequence.log, avoid setting ROS_LOG_DIR and ensure output="screen" in .launch.
    if not ONLY_ONE_SEQ_LOG:
        os.environ["ROS_LOG_DIR"] = exp_log_dir

    # ---- Launch ROS app via bash -c with tee so it prints to terminal and writes to file ----
    # We pass launch args inline; all stdout+stderr are piped to tee.
    cmd = (
        f"roslaunch {LAUNCH_FILE} "
        f"data_path:={DATASET_ROOT} "
        f"bag_file:={bag_pattern} "
        f"autorun:=1 "
        f"exp_log_dir:={exp_log_dir} "
        f"2>&1 | tee '{log_path}'"
    )

    launch_proc = subprocess.Popen(
        ["bash", "-c", cmd],
        preexec_fn=os.setsid
    )
    child_procs.append(launch_proc)

    # ---- Monitor /lastcloud with timeout (silent) ----
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

    # ---- Finish or abort gracefully ----
    try:
        if not got_lastcloud:
            kill_proc_group_safely(launch_proc, signal.SIGTERM)
            try:
                launch_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_proc_group_safely(launch_proc, signal.SIGKILL)
        else:
            launch_proc.wait()
    finally:
        kill_proc_group_safely(monitor_proc, signal.SIGTERM)
        try:
            monitor_proc.wait(timeout=3)
        except Exception:
            pass

    print(f"Completed: {folder_name}")

def main():
    if not os.environ.get("ROS_MASTER_URI"):
        print("ERROR: ROS_MASTER_URI is not set. Please start roscore first.")
        return

    subfolders = list_sequence_folders(DATASET_ROOT)
    if not subfolders:
        print("No 'day' or 'night' subfolders found. Exiting.")
        return

    print(f"Found {len(subfolders)} sequences.")
    print("Logging mode: terminal output is mirrored to exp_log_dir/<seq>/sequence.log via tee.")

    for folder in subfolders:
        folder_path = os.path.join(DATASET_ROOT, folder)
        try:
            run_one_sequence(folder_path, folder)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] Sequence '{folder}' failed: {e}")

        child_procs.clear()
        time.sleep(DELAY_BETWEEN)

    print("\nAll folders processed. Exiting.")

if __name__ == "__main__":
    main()
