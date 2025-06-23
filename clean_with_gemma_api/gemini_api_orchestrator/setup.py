#!/usr/bin/env python3
"""
Gepeta Setup - הכנת AWS resources
"""

import boto3
import os
from botocore.exceptions import ClientError


def setup_aws_keypair():
    """יצירת Key Pair"""
    ec2_client = boto3.client('ec2', region_name='us-east-1')
    key_name = "gepeta-worker-key"

    try:
        ec2_client.describe_key_pairs(KeyNames=[key_name])
        print(f"✅ Key Pair {key_name} כבר קיים")
        return True
    except ClientError:
        print(f"🔑 יוצר Key Pair {key_name}")

        response = ec2_client.create_key_pair(KeyName=key_name)

        with open(f"{key_name}.pem", 'w') as f:
            f.write(response['KeyMaterial'])

        if os.name != 'nt':
            os.chmod(f"{key_name}.pem", 0o400)

        print(f"✅ נוצר {key_name}.pem")
        print("⚠️ שמור את הקובץ במקום בטוח!")
        return True


def check_s3_bucket():
    """בדיקת S3 bucket"""
    s3_client = boto3.client('s3')
    bucket_name = "gepeta-datasets"

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ S3 Bucket {bucket_name} קיים")
        return True
    except ClientError:
        print(f"❌ S3 Bucket {bucket_name} לא קיים - צריך ליצור אותו")
        return False


def main():
    print("🔧 Gepeta Setup")
    print("=" * 30)

    try:
        # בדיקת AWS
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS: {identity['Account']}")

        # הכנת Key Pair
        setup_aws_keypair()

        # בדיקת S3
        check_s3_bucket()

        print("\n✅ Setup הושלם!")

    except Exception as e:
        print(f"❌ שגיאה: {e}")


if __name__ == "__main__":
    main()