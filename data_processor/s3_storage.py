import json
from datetime import datetime, timezone

import boto3


class S3EventSink:
    def __init__(self, config: dict):
        self.bucket = config["raw_events_bucket"]
        self.prefix = config["raw_events_prefix"].rstrip("/")
        client_arguments = {"region_name": config.get("region", "us-east-1")}
        if config.get("endpoint_url"):
            client_arguments["endpoint_url"] = config["endpoint_url"]
        self.client = boto3.client("s3", **client_arguments)

    def put_event(self, event: dict) -> str:
        event_id = str(event["event_id"])
        timestamp = datetime.fromisoformat(
            str(event["timestamp"]).replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        partition = timestamp.astimezone(timezone.utc).strftime("date=%Y-%m-%d")
        key = f"{self.prefix}/{partition}/{event_id}.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(event, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        return key
