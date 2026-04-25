import kfp
client = kfp.Client(host="http://localhost:8080")
pipeline = client.upload_pipeline(
    pipeline_package_path="fraud_pipeline.yaml",
    pipeline_name="fraud-pipeline",
    description="IEEE CIS Fraud Detection pipeline"
)
print(f"✅ Uploaded! ID: {pipeline.pipeline_id}")
