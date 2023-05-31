from google.cloud import storage
import os

bucket_name = "alan--soul-project"
dl_dir = "sentiment-gcp"

os.makedirs(dl_dir, exist_ok=True)
storage_client = storage.Client.from_service_account_json("src/gcp-key.json")

bucket = storage_client.get_bucket(bucket_name)
print("Downloading model files from GCP")
config_blob = bucket.blob("sentiment/config.json")
model_blob = bucket.blob("sentiment/tf_model.h5")

config_blob.download_to_filename(f"{dl_dir}/config.json")
model_blob.download_to_filename(f"{dl_dir}/tf_model.h5")
print("Done!")
