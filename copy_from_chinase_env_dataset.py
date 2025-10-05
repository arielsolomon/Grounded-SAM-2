import os
import shutil
import random


# ----------------------------------------
import os
import shutil
import random

# ---------------- CONFIG ----------------
root = '/media/ariels/home2/Arg/data/china_env_dataset/'
dst = '/media/ariels/home2/Git/Grounded-SAM-2/datasets/chin_env_data_for_stats/'
images_path = root+"Images"          # path to the images folder
annotations_path = root+"Annotation" # path to the annotations folder
output_images_path = dst+"images"
output_labels_path = dst+"labels"
n = 300  # number of images/labels to copy per subfolder
# ----------------------------------------

# Create output directories if they don't exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# Collect all image-label pairs across all subfolders
all_pairs = []

subfolders = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

for subfolder in subfolders:
    img_subfolder_path = os.path.join(images_path, subfolder)
    label_subfolder_path = os.path.join(annotations_path, subfolder)

    # List all files in subfolder
    img_files = [f for f in os.listdir(img_subfolder_path) if os.path.isfile(os.path.join(img_subfolder_path, f))]
    label_files = [f for f in os.listdir(label_subfolder_path) if os.path.isfile(os.path.join(label_subfolder_path, f))]

    # Match files by basename
    img_basenames = {os.path.splitext(f)[0]: f for f in img_files}
    label_basenames = {os.path.splitext(f)[0]: f for f in label_files}
    common_basenames = list(set(img_basenames.keys()) & set(label_basenames.keys()))

    for base in common_basenames:
        all_pairs.append((subfolder, img_basenames[base], label_basenames[base]))

# Randomly select n pairs
if len(all_pairs) < n:
    print(f"Warning: Total available pairs ({len(all_pairs)}) < n ({n}). Copying all available pairs.")
    selected_pairs = all_pairs
else:
    selected_pairs = random.sample(all_pairs, n)

# Copy selected pairs to output directories with new names
for subfolder, img_file, label_file in selected_pairs:
    new_img_name = f"{subfolder}_{img_file}"
    new_label_name = f"{subfolder}_{label_file}"

    shutil.copy2(os.path.join(images_path, subfolder, img_file), os.path.join(output_images_path, new_img_name))
    shutil.copy2(os.path.join(annotations_path, subfolder, label_file), os.path.join(output_labels_path, new_label_name))

print("Done! Random images and labels copied with unique names.")

