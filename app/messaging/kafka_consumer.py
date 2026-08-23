"""
Kafka consumer for receiving messages from the backend.
Handles incoming tenant requests and routes them to the orchestrator.
"""

import json
import asyncio
from typing import Callable, Dict, Any, Optional
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
import logging

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Kafka consumer for AI runtime requests."""

    def __init__(self, brokers: list, consumer_group: str):
        """
        Initialize Kafka consumer.

        Args:
            brokers: List of Kafka broker addresses
            consumer_group: Kafka consumer group identifier
        """
        self.brokers = brokers
        self.consumer_group = consumer_group
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._running = False

    async def discover_topics(
        self, pattern: str = "*.conversation.request"
    ) -> list[str]:
        """
        Discover topics matching a pattern using consumer metadata.

        Args:
            pattern: Topic name pattern (supports wildcards)

        Returns:
            List of matching topic names
        """
        temp_consumer = None
        try:
            # Create a temporary consumer to get cluster metadata
            temp_consumer = AIOKafkaConsumer(
                bootstrap_servers=self.brokers,
                group_id=f"{self.consumer_group}-discovery",
            )
            await temp_consumer.start()

            # Get all topics from the consumer's metadata
            # The topics() method is async and returns a set of all available topics
            all_topics = list(await temp_consumer.topics())

            await temp_consumer.stop()
            temp_consumer = None

            # Filter topics matching pattern
            # Simple pattern matching: convert *.conversation.request to regex
            import re

            pattern_regex = pattern.replace(".", r"\.").replace("*", ".*")
            matching_topics = [
                topic for topic in all_topics if re.match(pattern_regex, topic)
            ]

            logger.info(
                f"Discovered {len(matching_topics)} topics matching pattern {pattern}"
            )
            return matching_topics

        except Exception as e:
            logger.error(f"Error discovering topics: {e}", exc_info=True)
            # Ensure consumer is closed even on error
            if temp_consumer:
                try:
                    await temp_consumer.stop()
                except Exception:
                    pass
            return []

    async def start_consuming(
        self,
        topics: list[str],
        message_handler: Callable[[Dict[str, Any]], None],
        topic_pattern: str = "*.conversation.request",
    ):
        """
        Start consuming messages from Kafka topics.

        Args:
            topics: Initial list of Kafka topic names to subscribe to
            message_handler: Async callback function to handle incoming messages
            topic_pattern: Pattern for discovering new topics
        """
        try:
            logger.info(f"Starting Kafka consumer for topics: {topics}")
            # If no topics provided, try to discover them
            if not topics:
                logger.info(f"Discovering topics matching pattern: {topic_pattern}")
                topics = await self.discover_topics(topic_pattern)

            # If still no topics, start periodic discovery to check for new topics
            if not topics:
                logger.warning(
                    f"No topics found matching pattern {topic_pattern}. "
                    "Will retry every 5 seconds. Consumer will start when topics are discovered."
                )
                # Start a background task to periodically check for topics and restart consumer
                asyncio.create_task(
                    self._periodic_topic_discovery_and_restart(
                        topic_pattern, message_handler, interval=5
                    )
                )
                # Return early - consumer will start when topics are discovered
                return

            self.consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.brokers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                enable_auto_commit=True,
                auto_offset_reset="latest",
            )
            await self.consumer.start()
            self._running = True
            logger.info(f"Started Kafka consumer for topics: {topics}")
            logger.info(
                f"Consumer is now listening for messages. "
                f"Current offset strategy: 'latest' (only new messages will be consumed)"
            )

            # Start periodic topic refresh in background
            asyncio.create_task(
                self._periodic_topic_refresh(topic_pattern, message_handler)
            )

            # Start a heartbeat logger to show consumer is alive
            asyncio.create_task(self._log_consumer_heartbeat())

            async for message in self.consumer:
                if not self._running:
                    break
                try:
                    logger.debug(
                        f"Received message from topic {message.topic}, partition {message.partition}, offset {message.offset}"
                    )

                    # Extract tenant_id from topic name (format: {tenant_id}.conversation.request)
                    topic_parts = message.topic.split(".")
                    tenant_id = topic_parts[0] if len(topic_parts) > 0 else None

                    message_data = {
                        "tenant_id": tenant_id,
                        "topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset,
                        "key": message.key.decode("utf-8") if message.key else None,
                        "value": message.value,
                        # "headers": {
                        #     k.decode("utf-8"): v.decode("utf-8")
                        #     for k, v in message.headers or []
                        # },
                    }

                    logger.info(
                        f"Processing message from tenant {tenant_id}, conversation {message_data.get('value', {}).get('conversation_id', 'unknown')}"
                    )
                    await message_handler(message_data)
                    logger.debug(
                        f"Successfully processed message at offset {message.offset}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing message at offset {message.offset}: {e}",
                        exc_info=True,
                    )

        except KafkaError as e:
            logger.error(f"Kafka consumer error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in consumer: {e}", exc_info=True)
            raise

    async def _periodic_topic_discovery_and_restart(
        self, pattern: str, message_handler: Callable, interval: int = 10
    ):
        """Periodically discover new topics and restart consumer if new topics found."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                new_topics = await self.discover_topics(pattern)

                if new_topics:
                    # Check if consumer is running
                    if not self.consumer or not self._running:
                        # Restart consumer with new topics
                        logger.info(
                            f"Found {len(new_topics)} topics. Starting consumer..."
                        )
                        await self.start_consuming(new_topics, message_handler, pattern)
                        break  # Exit this task as consumer is now running
                    else:
                        # Check if we have new topics not in current subscription
                        try:
                            current_topics = set(self.consumer.subscription())
                            new_topics_set = set(new_topics)
                            if new_topics_set - current_topics:
                                logger.info(
                                    f"New topics discovered: {new_topics_set - current_topics}. "
                                    "Restarting consumer to subscribe to new topics..."
                                )
                                # Stop current consumer
                                await self.consumer.stop()
                                self.consumer = None
                                # Restart with all topics
                                await self.start_consuming(
                                    new_topics, message_handler, pattern
                                )
                                break
                        except Exception as e:
                            logger.debug(f"Could not check subscription: {e}")
            except Exception as e:
                logger.error(f"Error in periodic topic discovery: {e}")

    async def _periodic_topic_refresh(
        self, pattern: str, message_handler: Callable, interval: int = 60
    ):
        """Periodically refresh topic subscriptions."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                new_topics = await self.discover_topics(pattern)
                if new_topics and self.consumer:
                    try:
                        current_topics = set(self.consumer.subscription())
                        new_topics_set = set(new_topics)
                        if new_topics_set != current_topics:
                            logger.info(
                                f"New topics discovered. Current: {current_topics}, New: {new_topics_set}. "
                                "Restarting consumer..."
                            )
                            # Stop and restart consumer with new topics
                            await self.consumer.stop()
                            self.consumer = None
                            await self.start_consuming(
                                new_topics, message_handler, pattern
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Could not refresh topics: {e}")
            except Exception as e:
                logger.error(f"Error in periodic topic refresh: {e}")

    async def _log_consumer_heartbeat(self, interval: int = 60):
        """Periodically log that consumer is alive and waiting for messages."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self.consumer and self._running:
                    logger.debug(
                        f"Consumer is alive and waiting for messages on topics: {self.consumer.subscription()}"
                    )
            except Exception:
                pass

    async def stop_consuming(self):
        """Stop consuming messages and clean up resources."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Stopped Kafka consumer")
