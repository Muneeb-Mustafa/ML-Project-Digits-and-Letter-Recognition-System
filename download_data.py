import os
import shutil
import urllib.request
import zipfile
import sys
import time

# URL for the EMNIST dataset (provided by NIST)
URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
FILENAME = "emnist.zip"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading {FILENAME}: {percent}% ({progress_size / (1024*1024):.1f} MB, {speed} KB/s)")
    sys.stdout.flush()

def download_emnist():
    # Define cache directory (where the 'emnist' python package looks for files)
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "emnist")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    zip_path = os.path.join(cache_dir, FILENAME)
    
    print(f"Target Directory: {cache_dir}")
    print(f"Downloading from: {URL}")
    print("This file is approximately 535 MB. Please be patient.")

    try:
        urllib.request.urlretrieve(URL, zip_path, reporthook)
        print("\nDownload complete!")
        
        print("Verifying zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # We don't need to extract everything, the emnist package extracts what it needs from the zip
                # But we should check if it's valid
                print(f"Zip file is valid. Contains {len(zip_ref.namelist())} files.")
        except zipfile.BadZipFile:
            print("\nERROR: Downloaded file is corrupted (Bad Zip).")
            print("Please try running this script again or download manually.")
            os.remove(zip_path)
            return False

        print("\nSUCCESS: EMNIST dataset set up correctly.")
        print("You can now run 'python train_model.py'.")
        return True

    except Exception as e:
        print(f"\n\nERROR during download: {e}")
        print("-" * 50)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"1. Download the file from: {URL}")
        print(f"2. Rename it to '{FILENAME}'")
        print(f"3. Place it in this folder: {cache_dir}")
        print("-" * 50)
        return False

if __name__ == "__main__":
    download_emnist()
