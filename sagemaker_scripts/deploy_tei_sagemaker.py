#!/usr/bin/env python3
"""
Deploy Snowflake Arctic Embed model using the official HuggingFace TEI Container for SageMaker.
Based on: https://huggingface.co/blog/sagemaker-huggingface-embedding
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DELOY

# Setup SageMaker session
sess = sagemaker.Session()

# Get or create default bucket
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    sagemaker_session_bucket = sess.default_bucket()

# Get execution role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")

# Helper function to retrieve the correct image URI based on instance type
def get_image_uri(instance_type):
    """Get the appropriate TEI container image based on instance type (GPU vs CPU)"""
    key = "huggingface-tei" if instance_type.startswith("ml.g") or instance_type.startswith("ml.p") else "huggingface-tei-cpu"
    return get_huggingface_llm_image_uri(key)

instance_type = "ml.g5.xlarge"  # A10G GPU with 24GB (needed for large model)

for MODEL in MODELS_DELOY:
    # Create endpoint name from model name
    endpoint_name = MODEL.replace('/', '-').replace('.', '-') + '-endpoint'
    
    # Create HuggingFaceModel with the image uri
    emb_model = HuggingFaceModel(
        role=role,
        image_uri=get_image_uri(instance_type),
        env={'HF_MODEL_ID': MODEL},
    )

    print(f"\nDeploying model: {MODEL}")
    print(f"Endpoint name: {endpoint_name}")
    print("This can take ~5 minutes...")

    # Deploy model to an endpoint
    emb = emb_model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type=instance_type,
        wait=False,  # Don't wait for deployment to complete
    )
    
    print(f"✓ Deployment initiated for {endpoint_name}")
