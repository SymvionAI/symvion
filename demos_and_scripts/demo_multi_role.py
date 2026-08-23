import asyncio
import os
from symvion import Symvion, TenantConfig
from typing import Dict

# ---------------------------------------------------------
# Mock Database & Global State
# ---------------------------------------------------------
class Database:
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.agent_registrations: Dict[str, list] = {}
        
db = Database()

# ---------------------------------------------------------
# 1. SUPER ADMIN: Provisions infrastructure
# ---------------------------------------------------------
class SuperAdminAPI:
    @staticmethod
    def provision_new_client(tenant_id: str, env_vars: dict):
        print(f"[SUPER ADMIN] Provisioning new database space for Tenant: {tenant_id}")
        db.tenants[tenant_id] = TenantConfig(tenant_id=tenant_id, env_vars=env_vars)
        db.agent_registrations[tenant_id] = []

# ---------------------------------------------------------
# 2. TENANT ADMIN: Configures their specific AI logic
# ---------------------------------------------------------
class TenantAdminAPI:
    @staticmethod
    def add_custom_agent(tenant_id: str, agent_payload: dict):
        print(f"[TENANT ADMIN {tenant_id}] Registering specialized agent: {agent_payload['name']}")
        db.agent_registrations[tenant_id].append(agent_payload)

# ---------------------------------------------------------
# 3. BACKGROUND WORKER: Orchestrates AI / Host System / Kafka
# ---------------------------------------------------------
class AIWorker:
    def __init__(self):
        self.runtimes: Dict[str, Symvion] = {}

    def _get_runtime(self, tenant_id: str) -> Symvion:
        # Lazily load or rebuild the tenant's LangGraph based on DB state
        if tenant_id not in self.runtimes:
            config = db.tenants[tenant_id]
            runtime = Symvion(config=config)
            
            # Load all custom API-registered agents from DB
            for payload in db.agent_registrations[tenant_id]:
                runtime.register_agent(payload)
                
            self.runtimes[tenant_id] = runtime
            
        return self.runtimes[tenant_id]

    async def process_incoming_customer_message(self, tenant_id: str, session_id: str, message: str):
        print(f"\n--- [CUSTOMER] Message Received: '{message}' ---")
        runtime = self._get_runtime(tenant_id)
        
        # Execute Graph
        response = await runtime.chat(
            tenant=tenant_id,
            message=message,
            session_id=session_id
        )
        ai_reply = response["data"]

        # Intercept Escalations
        if "[ESCALATE]" in ai_reply:
            clean_reply = ai_reply.replace("[ESCALATE]", "").strip()
            print(f"[SYSTEM -> CUSTOMER] {clean_reply}")
            print(f"[SYSTEM -> KAFKA] Routing Session {session_id} to Human Agent Queue!")
            
            # Here: Send Kafka message to 'human_queue' topic
            HumanAgentPortal.receive_escalation(tenant_id, session_id)
        else:
            print(f"[SYSTEM -> CUSTOMER] {ai_reply}")

# ---------------------------------------------------------
# 4. HUMAN AGENT: Handles Escalated Tickets
# ---------------------------------------------------------
class HumanAgentPortal:
    @staticmethod
    def receive_escalation(tenant_id: str, session_id: str):
        print(f"[HUMAN AGENT PORTAL {tenant_id}] *DING* Support ticket opened for session: {session_id}")
        # Human agent takes over database session and chats directly with customer

# ---------------------------------------------------------
# INTEGRATION DEMONSTRATION
# ---------------------------------------------------------
async def main():
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    
    # 1. Super Admin onboard Acme Corp
    SuperAdminAPI.provision_new_client("acme_corp", {"OPENAI_KEY": "..."})
    
    # 2. Acme Corp's Tenant Admin configures a high-priority refund agent
    TenantAdminAPI.add_custom_agent("acme_corp", {
        "name": "refund_processor",
        "description": "Handles aggressive customers wanting refunds",
        "system_prompt": "You handle refunds. If they are extremely angry, you must output exactly: [ESCALATE] I am finding a human manager for you.",
        "input_schema": {}, "output_schema": {}, "tools": []
    })
    
    worker = AIWorker()

    # 3. Customer 1 explicitly needs the specialized AI agent
    await worker.process_incoming_customer_message(
        tenant_id="acme_corp",
        session_id="chat_123",
        message="I hate your product! I am so angry and want a refund!"
    )
    
    # 4. Customer 2 just says hello (gets defaulted by routing layer)
    await worker.process_incoming_customer_message(
        tenant_id="acme_corp",
        session_id="chat_456",
        message="Hi, I just want to say the product is great."
    )

if __name__ == "__main__":
    asyncio.run(main())
