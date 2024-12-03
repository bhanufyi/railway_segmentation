import os
from pathlib import Path
from google.cloud import storage

# Set the path to your service account key file
SERVICE_ACCOUNT_KEY_PATH = "./local-dev.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH

# Define the bucket and GCS folder paths
GCS_BUCKET_NAME = "rail-segmentation"  # Replace with your GCS bucket name
GCS_FOLDER = "weights/"  # The folder in GCS where the weights will be stored
LOCAL_FOLDER = "weights"
LOCAL_DOWNLOAD_FOLDER = "weights-from-gcs"


def upload_folder_to_gcs(local_folder, bucket_name, gcs_folder):
    """Uploads a local folder to a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Walk through the local folder and upload files
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = Path(root) / file
            blob_path = gcs_folder + str(local_path.relative_to(local_folder))
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            print(f"Uploaded {local_path} to {bucket_name}/{blob_path}")


def download_folder_from_gcs(bucket_name, gcs_folder, local_folder):
    """Downloads a folder from GCS to a local directory."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure local folder exists
    Path(local_folder).mkdir(parents=True, exist_ok=True)

    # List and download all blobs in the GCS folder
    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        relative_path = Path(blob.name).relative_to(gcs_folder)
        local_path = Path(local_folder) / relative_path

        # Ensure the local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        blob.download_to_filename(str(local_path))
        print(f"Downloaded {blob.name} to {local_path}")


def main():
    # Upload the weights folder to GCS
    print("Uploading weights folder to GCS...")
    upload_folder_to_gcs(LOCAL_FOLDER, GCS_BUCKET_NAME, GCS_FOLDER)

    # Download the folder back from GCS
    print("Downloading weights folder from GCS...")
    download_folder_from_gcs(GCS_BUCKET_NAME, GCS_FOLDER, LOCAL_DOWNLOAD_FOLDER)


if __name__ == "__main__":
    main()
