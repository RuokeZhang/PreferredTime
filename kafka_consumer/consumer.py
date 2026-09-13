import json
from pathlib import Path

import yaml
from kafka import KafkaConsumer

from data_processor.s3_storage import S3EventSink


class RawEventConsumer:
    def __init__(self, config: dict):
        kafka_config = config["kafka"]
        self.event_sink = S3EventSink(config["s3"])
        self.consumer = KafkaConsumer(
            kafka_config["topic"],
            bootstrap_servers=kafka_config["bootstrap_servers"],
            group_id=kafka_config["group_id"],
            auto_offset_reset=kafka_config["auto_offset_reset"],
            enable_auto_commit=False,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )

    def process_event(self, event: dict) -> str:
        required_fields = {"event_id", "user_id", "movie_id", "rating", "timestamp"}
        missing_fields = required_fields.difference(event)
        if missing_fields:
            raise ValueError(f"event is missing fields: {sorted(missing_fields)}")
        return self.event_sink.put_event(event)

    def start(self) -> None:
        try:
            for message in self.consumer:
                self.process_event(message.value)
                self.consumer.commit()
        finally:
            self.consumer.close()


def main() -> None:
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    RawEventConsumer(config).start()


if __name__ == "__main__":
    main()
