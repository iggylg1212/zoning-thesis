import boto3
import s3fs
import os

S3 = boto3.client("s3")
S3_PATH_NAME = "thesis1212"
S3_PATH = f"s3://{S3_PATH_NAME}/"

S3FS = s3fs.S3FileSystem()