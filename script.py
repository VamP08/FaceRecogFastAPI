import pandas as pd
import requests
import time
import os
from typing import List, Tuple

# --- ⚙️ Configuration ---
# 1. The URL of your API endpoint for uploading
API_URL = "https://facehrms.techvizor.in//upload"

# 2. The path to the folder containing your images (e.g., "profile")
IMAGE_FOLDER_PATH = "profile" 

# 3. The name of your Excel or CSV file
# IMPORTANT: Your file must have these exact column names: 
# id, name, image, member_code, member_status
DATA_FILE = "oliv-member-with-status.csv" # Change this to your filename

# 4. The file extension of your images (e.g., ".jpg", ".png")
ALLOWED_EXTENSIONS = [".jpg", ".JPG", ".png", ".PNG"] # Make sure this matches your image files

# --- Script Logic ---

def upload_all_employees():
    """
    Reads a data file, filters for active members, and uploads their
    corresponding images and data to the API.
    """
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: The data file '{DATA_FILE}' was not found.")
        return
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"❌ Error: The image folder '{IMAGE_FOLDER_PATH}' was not found.")
        return

    try:
        # Read the data file, ensuring all columns are treated as strings
        if DATA_FILE.endswith('.xlsx'):
            df = pd.read_excel(DATA_FILE, dtype=str)
        elif DATA_FILE.endswith('.csv'):
            df = pd.read_csv(DATA_FILE, dtype=str)
        else:
            print(f"❌ Error: Unsupported file format. Please use .xlsx or .csv")
            return
        print(f"✅ Successfully loaded {len(df)} total records from '{DATA_FILE}'.")
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return

    # Filter for only active members
    active_df = df[df['member_status'].str.upper() == 'ACTIVE'].copy()
    print(f"Found {len(active_df)} active members to process.")

    for index, row in active_df.iterrows():
        # --- Read data from the required columns ---
        employee_id = str(row["id"])
        name = str(row["name"])
        member_code = str(row["member_code"])
        image_filename_base = str(row["image"])
        
        print(f"\nProcessing {name} (ID: {employee_id}, Member: {member_code})")

        # --- Construct the image path and read the file ---
        image_filename = None
        local_image_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_filename = image_filename_base + ext
            potential_path = os.path.join(IMAGE_FOLDER_PATH, potential_filename)
            if os.path.exists(potential_path):
                image_filename = potential_filename
                local_image_path = potential_path
                break # Stop searching once the file is found

        if not local_image_path:
            print(f"  ❌ WARNING: Image for '{image_filename_base}' not found with any extension. Skipping {name}.")
            continue
            
        try:
            with open(local_image_path, 'rb') as f:
                image_content = f.read()
        except Exception as e:
            print(f"  ❌ WARNING: Could not read image file. Error: {e}")
            continue

        # --- Prepare the data for the API request ---
        payload = {
            'id': employee_id,
            'name': name,
            'member_code': member_code
        }
        
        files = {
            'pictures': (image_filename, image_content, 'image/jpeg')
        }

        # --- Send the POST request to the API ---
        try:
            print(f"  🚀 Uploading data for {name}...")
            upload_response = requests.post(API_URL, data=payload, files=files, timeout=60)
            upload_response.raise_for_status()
            
            print(f"  ✅ SUCCESS! Server response: {upload_response.json().get('MESSAGE')}")

        except requests.exceptions.RequestException as e:
            print(f"  ❌ FAILED to upload data for {name}. Error: {e}")
            if e.response is not None:
                print(f"  Server details: {e.response.text}")
        
        time.sleep(0.5) # A small delay to avoid overwhelming the server

    print("\n--- Script finished ---")


if __name__ == "__main__":
    upload_all_employees()