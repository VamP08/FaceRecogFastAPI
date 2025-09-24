import pandas as pd
import requests
import time
import os

# --- Configuration ---
# 1. The URL of your API endpoint for uploading
API_URL = "http://127.0.0.1:8000/upload"

# 2. The path to the folder containing your images
IMAGE_FOLDER_PATH = "images" 

# 3. The name of your Excel file
EXCEL_FILE = "usersolivenew.xlsx" 

# 4. The names of the columns in your Excel file
NAME_COLUMN = "name"
IMAGE_KEY_COLUMN = "image" # This column's value is used as the filename

# 5. The file extension of your images (e.g., ".JPG", ".jpg", ".png")
IMAGE_EXTENSION = ".JPG"

# 6. Starting employee ID to generate for each person
STARTING_ID = 1

# --- Script Logic ---

def upload_data():
    """
    Reads an Excel file and uploads corresponding local images to the API.
    """
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Error: The Excel file '{EXCEL_FILE}' was not found.")
        return
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"❌ Error: The image folder '{IMAGE_FOLDER_PATH}' was not found.")
        return

    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"✅ Successfully loaded {len(df)} records from '{EXCEL_FILE}'.")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    for index, row in df.iterrows():
        name = str(row[NAME_COLUMN])
        image_key = str(row[IMAGE_KEY_COLUMN])
        employee_id = STARTING_ID + index
        
        print(f"\nProcessing record {index + 1}/{len(df)}: Name='{name}', Image Key='{image_key}'")

        # --- Step 1: Construct the local file path and read the image ---
        image_filename = image_key + IMAGE_EXTENSION
        local_image_path = os.path.join(IMAGE_FOLDER_PATH, image_filename)

        try:
            print(f"   Reading local file: {local_image_path}")
            with open(local_image_path, 'rb') as f:
                image_content = f.read()
            print("   ✅ Image file read successfully.")
        except FileNotFoundError:
            print(f"   ❌ FAILED: Image file not found at '{local_image_path}'")
            continue # Skip to the next person
        except Exception as e:
            print(f"   ❌ FAILED: Could not read image file. Error: {e}")
            continue

        # --- Step 2: Prepare the data for upload ---
        payload = {
            'name': name,
            'id': str(employee_id)
        }
        
        files = {
            # Use the actual filename for the upload
            'pictures': (image_filename, image_content, 'image/jpeg')
        }

        # --- Step 3: Send the POST request to your API ---
        try:
            print(f"   Uploading data for {name} to the API...")
            upload_response = requests.post(API_URL, data=payload, files=files, timeout=30)
            upload_response.raise_for_status()
            
            print(f"   ✅ SUCCESS! Server response: {upload_response.json()}")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ FAILED to upload data for {name}. Error: {e}")
            if e.response is not None:
                print(f"   Server details: {e.response.text}")
        
        time.sleep(0.5) 

    print("\n--- Script finished ---")


if __name__ == "__main__":
    upload_data()