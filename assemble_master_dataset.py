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

 # Round angle values to 4 decimal places in the master dataset
master['angle'] = master['angle'].round(4)
master.to_csv(OUTPUT_CSV, index=False)
print(f"Master dataset written to {OUTPUT_CSV}")

# 6. Pivot to get passive and active angles per pose
pivot = master.pivot_table(
    index=["participant_id", "pose_name"],
    columns="variant",
    values="angle"
).reset_index()
# Rename columns for clarity
pivot = pivot.rename(columns={"passive": "passive_angle", "active": "active_angle"})

# 6a. Manually fill missing passive angles for specific participant–pose combos
pivot.loc[
    (pivot.participant_id == 1) & (pivot.pose_name == "side_splits"),
    "passive_angle"
] = 134.01
pivot.loc[
    (pivot.participant_id == 2) & (pivot.pose_name == "side_splits"),
    "passive_angle"
] = 116.66
pivot.loc[
    (pivot.participant_id == 3) & (pivot.pose_name == "side_splits"),
    "passive_angle"
] = 135.89
pivot.loc[
    (pivot.participant_id == 3) & (pivot.pose_name == "front_splits left"),
    "passive_angle"
] = 156.91
pivot.loc[
    (pivot.participant_id == 3) & (pivot.pose_name == "front_splits right"),
    "passive_angle"
] = 156.59
pivot.loc[
    (pivot.participant_id == 4) & (pivot.pose_name == "side_splits"),
    "passive_angle"
] = 100.28
pivot.loc[
    (pivot.participant_id == 8) & (pivot.pose_name == "side_splits"),
    "passive_angle"
] = 78.80
# Compute normalized difference
pivot["norm_diff"] = (pivot["passive_angle"] - pivot["active_angle"]) / pivot["passive_angle"]

# Save pivoted dataset
PIVOT_CSV = "yoga_pivoted_dataset.csv"
 # Round passive_angle, active_angle, and norm_diff to 4 decimal places
pivot[['passive_angle','active_angle','norm_diff']] = pivot[['passive_angle','active_angle','norm_diff']].round(4)
pivot.to_csv(PIVOT_CSV, index=False)
print(f"Pivoted dataset written to {PIVOT_CSV}")

# 7. Compute MWI per participant
mwi = pivot.groupby("participant_id")["norm_diff"].mean().reset_index(name="MWI")
MWI_CSV = "yoga_MWI.csv"
 # Round MWI values to 4 decimal places
mwi['MWI'] = mwi['MWI'].round(4)
mwi.to_csv(MWI_CSV, index=False)
print(f"MWI per participant written to {MWI_CSV}")