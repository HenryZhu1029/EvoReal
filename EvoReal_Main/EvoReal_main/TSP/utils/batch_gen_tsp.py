
import os
import subprocess
import tempfile
import logging
import shutil
import sys

def get_console_logger(name=None):
    logger = logging.getLogger(name or f"console_logger_{os.getpid()}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    # Avoid repeated adding handler
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def save_tsplib_format(coords, filename, scale=10000):
    n = coords.shape[0]
    with open(filename, "w") as f:
        f.write("NAME: test_tsp\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i+1} {int(round(float(x)*scale))} {int(round(float(y)*scale))}\n")
        f.write("EOF\n")

def read_tour(sol_file):
    with open(sol_file, "r") as f:
        lines = f.readlines()
    tour = []
    for line in lines[1:]:  # Skip the first row (points)
        line = line.strip()
        if not line:
            continue
        for node in line.split():
            tour.append(int(node)+1)
    return tour


def batch_gen_tsp(batch_coords, save_path, concorde_path, logger, solved_counter=None, max_valid=None):
    for i, coords in enumerate(batch_coords):
        # ---- Global termination judgment ----
        if solved_counter is not None and max_valid is not None:
            if solved_counter.value >= max_valid:
                logger.info(f"Process {os.getpid()}: Early exit, global valid count reached {max_valid}.")
                break

        with tempfile.TemporaryDirectory() as tempdir:
            tsp_file = os.path.join(tempdir, "problem.tsp")
            tour_file = os.path.join(tempdir, "problem.tour")
            save_tsplib_format(coords, tsp_file)

            try:
                subprocess.run(
                    [concorde_path, "-o", tour_file, tsp_file],
                    timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL # time out for each problem is 5 seconds
                )
                if not os.path.exists(tour_file):
                    logger.warning(f"Problem {i}: Tour file not created (timeout or Concorde failed). Skipped.")
                    continue
            except subprocess.TimeoutExpired:
                logger.warning(f"Problem {i}: Timeout (>2s). Skipped.")
                continue
            except Exception as e:
                logger.warning(f"Problem {i}: Runtime error {e}. Skipped.")
                continue

            tour = read_tour(tour_file)
            if tour is None or not (coords.shape[0] <= len(tour) <= coords.shape[0] + 1):
                logger.warning(
                    f"Problem {i}: Invalid tour. Tour length {len(tour) if tour else 'None'} != n or n+1 (n={coords.shape[0]}). Skipped."
                )
                continue

        # ---- Standardized tour for n points ----
        if len(tour) == coords.shape[0] + 1 and tour[-1] == tour[0]:
            tour = tour[:-1]

        # ---- Global successful counts ----
        valid_this = False
        if solved_counter is not None and max_valid is not None:

            if solved_counter.value < max_valid:
                solved_counter.value += 1
                valid_this = True
            else:
                logger.info(f"Process {os.getpid()}: Reached max_valid={max_valid}, skipping write.")
                break
        else:
            valid_this = True  # It does not affect in a single-process scenario

        # ---- write ----
        if valid_this:
            coord_flat = " ".join([f"{float(x):.12f} {float(y):.12f}" for x, y in coords])
            tour_flat = " ".join([str(t) for t in tour])
            with open(save_path, "a") as fout:
                fout.write(f"{coord_flat} output {tour_flat}\n")

        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Process {os.getpid()} - {i+1}/{len(batch_coords)} done, global_valid: {solved_counter.value if solved_counter else 'NA'}")

def detect_concorde():
    # Prioritize environment variable
    path = os.environ.get("CONCORDE_PATH")
    if path and os.path.exists(path):
        return path
    # Check the global command again
    if shutil.which("concorde") is not None:
        return "concorde"

    default_path = ""           # your path to concorde.exe
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError("Concorde not found. Please set CONCORDE_PATH or ensure it is in your PATH.")


