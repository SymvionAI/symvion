"""
FastAPI entry point for the AI Runtime service.
Handles HTTP endpoints and initializes the application.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI

from app.config import (
    KAFKA_BROKERS,
    KAFKA_CONSUMER_GROUP,
    APP_ENV,
)
from app.messaging.kafka_consumer import KafkaConsumer
from app.messaging.kafka_producer import KafkaProducer
from app.messaging.topics import get_response_topic
from app.runtime.orchestrator import Orchestrator
from app.runtime.graph_factory import GraphFactory
from app.runtime.memory.memory_store import MemoryStore

# Configure logging
logging.basicConfig(
    level=logging.INFO if APP_ENV == "production" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress verbose Kafka client logs (aiokafka, kafka-python)
logging.getLogger("aiokafka").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)
logging.getLogger("aiokafka.conn").setLevel(logging.ERROR)
logging.getLogger("aiokafka.consumer").setLevel(logging.WARNING)
logging.getLogger("aiokafka.producer").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global instances
kafka_consumer: KafkaConsumer | None = None
kafka_producer: KafkaProducer | None = None
orchestrators: Dict[str, Orchestrator] = {}
graphs: Dict[str, Any] = {}  # Store compiled LangGraph instances
memory_stores: Dict[str, MemoryStore] = {}


def get_or_create_orchestrator(
    tenant_id: str,
    tenant_name: Optional[str] = None,
    allowed_agents: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
) -> Orchestrator:
    """
    Get or create an orchestrator instance for a tenant.

    Args:
        tenant_id: Unique tenant identifier
        tenant_name: Optional tenant name (falls back to tenant_id if not provided)
        allowed_agents: Optional list of allowed agent IDs (empty means all allowed)
        allowed_tools: Optional list of allowed tool IDs (empty means all allowed)

    Returns:
        Orchestrator instance for the tenant
    """
    if tenant_id not in orchestrators:
        # Get or create memory store
        if tenant_id not in memory_stores:
            memory_stores[tenant_id] = MemoryStore(tenant_id)

        # Tenant config with data from backend
        tenant_config = {
            "tenant_name": tenant_name or tenant_id,
            "allowed_agents": allowed_agents or [],  # Empty means all allowed
            "allowed_tools": allowed_tools or [],  # Empty means all allowed
        }

        orchestrators[tenant_id] = Orchestrator(
            tenant_id=tenant_id,
            config=tenant_config,
            memory_store=memory_stores[tenant_id],
        )
        logger.info(
            f"Created orchestrator for tenant: {tenant_id} "
            f"(name: {tenant_config['tenant_name']}, "
            f"agents: {len(tenant_config['allowed_agents'])}, "
            f"tools: {len(tenant_config['allowed_tools'])})"
        )
    else:
        # Update config if provided and different
        updated = False
        if tenant_name and tenant_name != orchestrators[tenant_id].tenant_name:
            orchestrators[tenant_id].tenant_name = tenant_name
            orchestrators[tenant_id].config["tenant_name"] = tenant_name
            updated = True

        if allowed_agents is not None:
            current_agents = orchestrators[tenant_id].allowed_agents
            if set(allowed_agents) != set(current_agents):
                orchestrators[tenant_id].allowed_agents = allowed_agents
                orchestrators[tenant_id].config["allowed_agents"] = allowed_agents
                updated = True

        if allowed_tools is not None:
            current_tools = orchestrators[tenant_id].allowed_tools
            if set(allowed_tools) != set(current_tools):
                orchestrators[tenant_id].allowed_tools = allowed_tools
                orchestrators[tenant_id].config["allowed_tools"] = allowed_tools
                updated = True

        if updated:
            orchestrators[tenant_id].system_prompt = orchestrators[
                tenant_id
            ]._build_system_prompt()
            logger.info(
                f"Updated config for tenant {tenant_id}: "
                f"name={tenant_name or 'unchanged'}, "
                f"agents={len(allowed_agents) if allowed_agents else 'unchanged'}, "
                f"tools={len(allowed_tools) if allowed_tools else 'unchanged'}"
            )

    return orchestrators[tenant_id]


async def handle_kafka_message(message_data: Dict[str, Any]):
    """
    Handle incoming Kafka message from backend.

    Args:
        message_data: Message data from Kafka consumer
    """
    try:
        value = message_data.get("value", {})
        tenant_id = value.get("tenant_id") or message_data.get("tenant_id")
        conversation_id = value.get("conversation_id")
        user_message = value.get("message")
        context = value.get("context", {})

        # Debug: Log received message data
        logger.info(
            f"[KAFKA_HANDLER] Received Kafka message for tenant {tenant_id}, conversation {conversation_id}"
        )
        logger.info(
            f"[KAFKA_HANDLER] Context keys: {list(context.keys()) if context else 'None'}"
        )
        logger.info(
            f"[KAFKA_HANDLER] Full context: {json.dumps(context, indent=2) if context else 'None'}"
        )

        if context.get("documents"):
            logger.info(
                f"[KAFKA_HANDLER] Found {len(context.get('documents', []))} document(s) in context.documents"
            )
            for idx, doc in enumerate(context.get("documents", [])):
                logger.info(
                    f"[KAFKA_HANDLER] Document {idx + 1}: id={doc.get('id')}, fileName={doc.get('fileName')}, "
                    f"mimeType={doc.get('mimeType')}, downloadUrl={doc.get('downloadUrl')}, "
                    f"cloudinaryUrl={doc.get('cloudinaryUrl')}"
                )
        elif context.get("message_metadata"):
            msg_meta = context.get("message_metadata", {})
            if msg_meta.get("documents"):
                logger.info(
                    f"[KAFKA_HANDLER] Found {len(msg_meta.get('documents', []))} document(s) in message_metadata"
                )
            else:
                logger.warning(
                    f"[KAFKA_HANDLER] message_metadata exists but has no documents. Keys: {list(msg_meta.keys()) if msg_meta else 'None'}"
                )
        else:
            logger.warning(
                f"[KAFKA_HANDLER] No documents found in context or message_metadata"
            )

        # Extract tenant_name from context or value if available
        tenant_name = (
            context.get("tenant_name")
            or value.get("tenant_name")
            or message_data.get("tenant_name")
        )

        # Extract allowed_agents and allowed_tools from context or value
        allowed_agents = (
            context.get("allowed_agents") or value.get("allowed_agents") or []
        )
        allowed_tools = context.get("allowed_tools") or value.get("allowed_tools") or []

        print(
            f"Processing message for tenant {tenant_id}, conversation {conversation_id} {user_message}"
        )

        if not tenant_id or not conversation_id or not user_message:
            logger.warning(f"Invalid message format: {message_data}")
            return

        logger.info(
            f"Processing message for tenant {tenant_id}, conversation {conversation_id} "
            f"(agents: {len(allowed_agents)}, tools: {len(allowed_tools)})"
        )

        # Use GraphFactory if agents are configured, otherwise fall back to Orchestrator
        if allowed_agents and len(allowed_agents) > 0:
            # Extract auth token from context
            auth_token = context.get("auth_token")

            # Get or create graph for tenant
            graph_key = f"{tenant_id}:{':'.join(sorted(allowed_agents))}"
            if graph_key not in graphs:
                tenant_config = {
                    "tenant_name": tenant_name or tenant_id,
                    "allowed_agents": allowed_agents,
                    "allowed_tools": allowed_tools or [],
                    "agent_config": {
                        "claims": {
                            "auth_token": auth_token,  # Pass auth token to claims agent
                        },
                    },
                }
                graph = GraphFactory.build_graph(
                    tenant_id=tenant_id,
                    allowed_agents=allowed_agents,
                    allowed_tools=allowed_tools or [],
                    tenant_config=tenant_config,
                )
                graphs[graph_key] = graph
                logger.info(
                    f"Created graph for tenant {tenant_id} with agents: {allowed_agents}"
                )

            graph = graphs[graph_key]

            # Get conversation history from memory store
            if tenant_id not in memory_stores:
                memory_stores[tenant_id] = MemoryStore(tenant_id)
            memory_store = memory_stores[tenant_id]
            conversation_memory = memory_store.get_conversation_memory(conversation_id)
            message_history = (
                conversation_memory.get("history", []) if conversation_memory else []
            )

            # Extract documents from context (backend now sends documents directly in context)
            documents = context.get("documents", [])

            # If not in context, try message_metadata as fallback
            if not documents:
                message_metadata = (
                    context.get("message_metadata")
                    or value.get("message_metadata")
                    or {}
                )
                if message_metadata.get("documents"):
                    documents = message_metadata["documents"]

            # Ensure documents are in context for agent access
            if documents:
                context = context or {}
                context["documents"] = documents
                logger.info(f"Found {len(documents)} document(s) to process")

            # Persist document_analysis from previous turn so claims/other agents can use it on follow-up messages
            if conversation_memory and conversation_memory.get("document_analysis") and not (context or {}).get("document_analysis"):
                context = context or {}
                context["document_analysis"] = conversation_memory["document_analysis"]

            # Create initial state
            initial_state = GraphFactory.create_initial_state(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                user_message=user_message,
                context=context,
                message_history=message_history,
            )

            # Invoke graph with error handling
            try:
                final_state = await graph.ainvoke(initial_state)

                # Get the final response and agent type
                agent_response = final_state.get("agent_response", "I'm here to help.")
                agent_type = final_state.get("agent_type", "general")

                # Detect AI-requested escalation ([ESCALATE] prefix)
                escalation_requested = "[ESCALATE]" in (agent_response or "")
                if escalation_requested:
                    agent_response = (
                        agent_response.replace("[ESCALATE]", "")
                        .strip()
                        .strip("\n")
                        or "I'm connecting you with our team. A team member will be with you shortly."
                    )

                # Update conversation memory (history + persisted context for next turn)
                if not conversation_memory:
                    conversation_memory = {"history": []}
                conversation_memory["history"].append(
                    {"role": "user", "content": user_message}
                )
                conversation_memory["history"].append(
                    {"role": "assistant", "content": agent_response}
                )
                # Persist document_analysis so claims agent has it on follow-up turns (e.g. "Yes", "It's in the document")
                final_context = final_state.get("context") or {}
                if final_context.get("document_analysis"):
                    conversation_memory["document_analysis"] = final_context["document_analysis"]
                memory_store.update_conversation_memory(
                    conversation_id, conversation_memory
                )

                # Build response (metadata uses camelCase for backend)
                response = {
                    "content": agent_response,
                    "metadata": {
                        "model": os.getenv("LLM_MODEL", "gpt-4"),
                        "tenant_id": tenant_id,
                        "conversation_id": conversation_id,
                        "agent_type": agent_type,
                        "escalationRequested": escalation_requested,
                    },
                }
            except Exception as graph_error:
                logger.error(
                    f"Error invoking graph for conversation {conversation_id}: {graph_error}",
                    exc_info=True,
                )
                # Send error response to prevent blocking
                response = {
                    "content": "I apologize, but I encountered an error processing your message. Please try again.",
                    "metadata": {
                        "model": os.getenv("LLM_MODEL", "gpt-4"),
                        "tenant_id": tenant_id,
                        "conversation_id": conversation_id,
                        "agent_type": "error",
                        "error": str(graph_error),
                    },
                }
        else:
            # Fall back to orchestrator if no agents configured
            orchestrator = get_or_create_orchestrator(
                tenant_id,
                tenant_name=tenant_name,
                allowed_agents=allowed_agents,
                allowed_tools=allowed_tools,
            )

            # Process message through orchestrator
            response = await orchestrator.process_message(
                message=user_message,
                conversation_id=conversation_id,
                context=context,
            )

        # Send response back to backend
        if kafka_producer:
            await kafka_producer.send_response(
                topic=get_response_topic(),
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                response=response,
            )
            logger.info(
                f"Sent response for conversation {conversation_id} to topic {get_response_topic()}"
            )
        else:
            logger.error("Kafka producer not initialized")

    except Exception as e:
        logger.error(f"Error handling Kafka message: {e}", exc_info=True)
        # Always send an error response to prevent blocking other messages
        try:
            tenant_id = message_data.get("value", {}).get(
                "tenant_id"
            ) or message_data.get("tenant_id", "unknown")
            conversation_id = message_data.get("value", {}).get(
                "conversation_id", "unknown"
            )

            error_response = {
                "content": "I apologize, but I encountered an error processing your message. Please try again.",
                "metadata": {
                    "model": os.getenv("LLM_MODEL", "gpt-4"),
                    "tenant_id": tenant_id,
                    "conversation_id": conversation_id,
                    "agent_type": "error",
                    "error": str(e),
                },
            }

            if kafka_producer:
                await kafka_producer.send_response(
                    topic=get_response_topic(),
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    response=error_response,
                )
                logger.info(f"Sent error response for conversation {conversation_id}")
            else:
                logger.error(
                    "Kafka producer not initialized, cannot send error response"
                )
        except Exception as send_error:
            logger.error(f"Failed to send error response: {send_error}", exc_info=True)


async def start_kafka_consumption():
    """Start consuming messages from Kafka. Non-blocking: logs errors and returns so HTTP server can go live."""
    global kafka_consumer, kafka_producer

    try:
        # Initialize producer
        kafka_producer = KafkaProducer(brokers=KAFKA_BROKERS)
        await kafka_producer.start()

        # Initialize consumer
        kafka_consumer = KafkaConsumer(
            brokers=KAFKA_BROKERS, consumer_group=KAFKA_CONSUMER_GROUP
        )

        logger.info("Discovering Kafka topics...")
        topics = await kafka_consumer.discover_topics("*.conversation.request")

        if topics:
            logger.info(f"Found {len(topics)} existing topics: {topics}")
        else:
            logger.warning(
                "No topics discovered initially. Consumer will periodically check for new topics. "
                "Make sure topics follow the pattern: {tenant_id}.conversation.request"
            )

        logger.info(f"Starting Kafka consumer for {len(topics)} topics...")
        asyncio.create_task(
            kafka_consumer.start_consuming(
                topics, handle_kafka_message, topic_pattern="*.conversation.request"
            )
        )
        logger.info("Kafka consumption started successfully")
    except Exception as e:
        # Do not raise: allow the app to start so /health is available and orchestrators can mark the pod live.
        # Kafka will be retried on next deploy or manual restart once Kafka is available.
        logger.error(
            "Failed to start Kafka consumption; service will run but AI messages will not be processed until Kafka is available: %s",
            e,
            exc_info=True,
        )


async def stop_kafka_consumption():
    """Stop consuming messages from Kafka."""
    global kafka_consumer, kafka_producer

    if kafka_consumer:
        await kafka_consumer.stop_consuming()

    if kafka_producer:
        await kafka_producer.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events. Kafka starts in background so HTTP is not blocked."""
    # Startup: start Kafka without blocking (so /health is available even if Kafka is down)
    logger.info("Starting AI Runtime service...")
    asyncio.create_task(start_kafka_consumption())

    yield

    # Shutdown
    logger.info("Shutting down AI Runtime service...")
    await stop_kafka_consumption()
    logger.info("AI Runtime service stopped")


app = FastAPI(
    title="symvion AI Runtime",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "kafka_consumer": kafka_consumer is not None,
        "kafka_producer": kafka_producer is not None,
        "active_tenants": len(orchestrators),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "symvion AI Runtime", "version": "0.1.0"}
