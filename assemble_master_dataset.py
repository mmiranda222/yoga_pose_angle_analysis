import pandas as pd
import os
import itertools

# 1. Configuration
# ---------------

# Path to the CSV your analyzer already wrote out
INPUT_CSV = "yoga_pose_analysis_results.csv"
# Where you want the merged master CSV
OUTPUT_CSV = "yoga_master_dataset.csv"

# Map your 1–8 IDs to participant names (if you ever need them)
participant_names = {
    1: "marta_cardona",
    2: "camila_valencia",
    3: "camila_jaramillo",
    4: "jorbelys_perez",
    5: "milla_leipziger",
    6: "alicia_barrientos",
    7: "irene_alda",
    8: "javier_alda"
}

# Your file_code_map (same as in the analyzer)
file_code_map = {
    "LRP": ("lunge", "passive", "right"),
    "LRA": ("lunge", "active",  "right"),
    "LLP": ("lunge", "passive", "left"),
    "LLA": ("lunge", "active",  "left"),
    "CP":  ("cobra",  "passive", "none"),
    "CA":  ("cobra",  "active",  "none"),
    "HORP":("hip_opening", "passive", "right"),
    "HORA":("hip_opening", "active",  "right"),
    "HOLP":("hip_opening", "passive", "left"),
    "HOLA":("hip_opening", "active",  "left"),
    "FSRP":("front_splits", "passive", "right"),
    "FSRA":("front_splits", "active",  "right"),
    "FSLP":("front_splits", "passive", "left"),
    "FSLA":("front_splits", "active",  "left"),
    "SSA": ("side_splits", "active",  "none"),
    "SSP": ("side_splits", "passive", "none"),
}

# 2. Load analyzer output and parse filename codes
# ------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# Extract code and participant_id from filename, then map to labels
def parse_row(row):
    base = os.path.splitext(row.filename)[0]
    code, pid = base.split("_")
    pid = int(pid)
    if code in file_code_map:
        pose_type, variant, side = file_code_map[code]
        # build human-readable pose name
        pose_name = f"{pose_type} {side}" if side != "none" else pose_type
    else:
        pose_name, variant, pid = None, None, pid
    return pd.Series({
        "participant_id": pid,
        "pose_name": pose_name,
        "variant": variant,
        "angle": row.angle
    })

parsed = df.apply(parse_row, axis=1)

# 3. Build the full grid of participant×code
# ------------------------------------------

all_pids = list(participant_names.keys())
all_codes = list(file_code_map.keys())

grid = []
for pid, code in itertools.product(all_pids, all_codes):
    pose_type, variant, side = file_code_map[code]
    pose_name = f"{pose_type} {side}" if side!="none" else pose_type
    grid.append({
        "participant_id": pid,
        "pose_name": pose_name,
        "variant": variant
    })
full_df = pd.DataFrame(grid)

# 4. Merge and order
# ------------------

# left‐join to bring in angles (NaN where missing)
master = full_df.merge(parsed, on=["participant_id","pose_name","variant"], how="left")

# Define ordering: first by pose_name, then by participant_id,
# and within each (pose_name, pid) ensure active comes before passive
variant_order = {"active":0, "passive":1, None:2}
master["variant_rank"] = master["variant"].map(variant_order)

master = master.sort_values(
    by=["pose_name","participant_id","variant_rank"]
).drop(columns="variant_rank")

# 5. Save
# -------

master.to_csv(OUTPUT_CSV, index=False)
print(f"Master dataset written to {OUTPUT_CSV}")