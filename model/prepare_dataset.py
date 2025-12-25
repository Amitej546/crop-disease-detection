import os
import shutil

# Absolute path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct dataset paths (absolute)
RAW_ROOT = os.path.join(BASE_DIR, "..", "dataset", "PlantVillage_raw")
PROCESSED_ROOT = os.path.join(BASE_DIR, "..", "dataset", "processed")

RAW_ROOT = os.path.abspath(RAW_ROOT)
PROCESSED_ROOT = os.path.abspath(PROCESSED_ROOT)

print("📂 RAW DATASET PATH:", RAW_ROOT)

os.makedirs(PROCESSED_ROOT, exist_ok=True)

# Detect where class folders actually are
SOURCE_DIR = None

for item in os.listdir(RAW_ROOT):
    if "___" in item:
        SOURCE_DIR = RAW_ROOT
        break

if SOURCE_DIR is None:
    nested = os.path.join(RAW_ROOT, "PlantVillage")
    if os.path.exists(nested):
        SOURCE_DIR = nested

if SOURCE_DIR is None:
    raise Exception("❌ No valid PlantVillage class folders found")

print("📂 USING SOURCE:", SOURCE_DIR)

# Restructure dataset
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue
    if "___" not in folder:
        continue

    crop, disease = folder.split("___", 1)
    crop = crop.replace("__", "_")

    crop_dir = os.path.join(PROCESSED_ROOT, crop)
    disease_dir = os.path.join(crop_dir, disease)

    os.makedirs(disease_dir, exist_ok=True)

    for img in os.listdir(folder_path):
        src = os.path.join(folder_path, img)
        dst = os.path.join(disease_dir, img)

        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

print("✅ Dataset restructuring completed successfully.")
