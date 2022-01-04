import boto3
import s3fs
import io

S3FS = s3fs.S3FileSystem(anon=False)
S3 = boto3.client("s3")
S3_PATH_NAME = "thesis1212"
S3_PATH = f"s3://{S3_PATH_NAME}/"

S3FS = s3fs.S3FileSystem(anon=False)

def read_file(key):
    return io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=key)['Body'].read())

def write_csv(key, df):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, header=True, index=False)
    csv_buf.seek(0)
    S3.put_object(Bucket=S3_PATH_NAME, Body=csv_buf.getvalue(), Key=key)