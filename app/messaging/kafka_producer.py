"""
Kafka producer for sending messages to the backend.
Handles publishing responses and status updates.
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import logging

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Kafka producer for AI runtime responses."""

    def __init__(self, brokers: list):
        """
        Initialize Kafka producer.

        Args:
            brokers: List of Kafka broker addresses
        """
        self.brokers = brokers
        self.producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        """Start the Kafka producer."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self.producer.start()
            logger.info("Started Kafka producer")
        except KafkaError as e:
            logger.error(f"Failed to start Kafka producer: {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop the Kafka producer and clean up resources."""
        if self.producer:
            await self.producer.stop()
            logger.info("Stopped Kafka producer")

    async def send_response(
        self,
        topic: str,
        tenant_id: str,
        conversation_id: str,
        response: Dict[str, Any],
        key: Optional[str] = None,
    ):
        """
        Send a response message to Kafka.

        Args:
            topic: Kafka topic name
            tenant_id: Unique tenant identifier
            conversation_id: Unique conversation identifier
            response: Response data to send
            key: Optional message key (defaults to conversation_id)
        """
        if not self.producer:
            raise RuntimeError("Producer not started. Call start() first.")

        message = {
            "tenant_id": tenant_id,
            "conversation_id": conversation_id,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            await self.producer.send_and_wait(
                topic,
                value=message,
                key=(key or conversation_id).encode("utf-8"),
            )
            logger.debug(
                f"Sent response to topic {topic} for conversation {conversation_id}"
            )
        except KafkaError as e:
            logger.error(
                f"Failed to send response to topic {topic}: {e}", exc_info=True
            )
            raise
