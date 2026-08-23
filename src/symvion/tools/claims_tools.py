"""
Claims-related tools for the claims agent.
These tools interact with the backend claims API to perform claims operations.
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import httpx

from symvion.utils.security import hash_secret, redact_secret, validate_outbound_url

logger = logging.getLogger(__name__)


class ClaimsTools:
    """Collection of tools for claims operations."""

    def __init__(
        self,
        tenant_id: str,
        backend_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        *,
        allowed_outbound_hosts: Optional[list] = None,
    ):
        """
        Initialize claims tools.

        Args:
            tenant_id: Unique tenant identifier
            backend_url: Backend API URL (defaults to env var)
            auth_token: JWT token for authenticating with backend API
            allowed_outbound_hosts: Optional host allowlist for SSRF protection
        """
        self.tenant_id = tenant_id
        raw_url = backend_url or os.getenv(
            "BACKEND_URL", "http://localhost:3000"
        )
        self.backend_url = validate_outbound_url(
            raw_url,
            allowed_hosts=allowed_outbound_hosts,
            allow_localhost=True,
            require_https=False,
        )
        self.allowed_outbound_hosts = allowed_outbound_hosts
        self.base_url = f"{self.backend_url}/api/v1/claims"
        self.auth_token = auth_token
        # Do not follow redirects: prevents token exfiltration via open redirects.
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=False)

        logger.info(
            "Claims tools initialized for tenant %s (auth_token=%s)",
            self.tenant_id,
            redact_secret(self.auth_token),
        )

    def _get_headers(self, include_tenant: bool = True) -> Dict[str, str]:
        """Build headers with auth token and tenant ID."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if include_tenant:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers

    def _normalize_date(self, date_str: str) -> str:
        """
        Normalize date string to YYYY-MM-DD format.

        Args:
            date_str: Date string in various formats

        Returns:
            Date string in YYYY-MM-DD format

        Raises:
            ValueError: If date cannot be parsed
        """
        if not date_str:
            raise ValueError("Date string cannot be empty")

        try:
            # Try parsing as ISO format first
            if "T" in date_str:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                # Try parsing with common formats
                formats = [
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%d/%m/%Y",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%m-%d-%Y",
                ]
                date_obj = None
                for fmt in formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

                if date_obj is None:
                    # Last resort: try parsing as ISO date
                    date_obj = datetime.fromisoformat(date_str)

            return date_obj.strftime("%Y-%m-%d")
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Unable to parse date: {date_str}. Please use YYYY-MM-DD format (e.g., 2024-01-15). Error: {str(e)}"
            )

    async def submit_claim(
        self,
        policy_number: str,
        incident_date: str,
        incident_description: str,
        claim_type: str,
        estimated_amount: Optional[float] = None,
        contact_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a new insurance claim.

        Args:
            policy_number: Policy number associated with the claim
            incident_date: Date of the incident (ISO format: YYYY-MM-DD, e.g., "2024-01-15"). REQUIRED.
            incident_description: Description of what happened
            claim_type: Type of claim (e.g., "auto", "property", "health", "life")
            estimated_amount: Optional estimated claim amount
            contact_info: Optional contact information (phone, email, etc.)

        Returns:
            Dict containing claim ID and status
        """
        try:
            # Validate required parameters
            if not incident_date:
                raise ValueError("incident_date is required and cannot be empty")
            
            # Normalize incident_date to YYYY-MM-DD format
            normalized_date = self._normalize_date(incident_date)

            payload = {
                "policy_number": policy_number,
                "incident_date": normalized_date,
                "incident_description": incident_description,
                "claim_type": claim_type,
                "tenant_id": self.tenant_id,
            }
            if estimated_amount:
                payload["estimated_amount"] = estimated_amount
            if contact_info:
                payload["contact_info"] = contact_info

            response = await self.client.post(
                f"{self.base_url}",
                json=payload,
                headers=self._get_headers(include_tenant=False),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Successfully submitted claim for tenant {self.tenant_id}: {result.get('id')}"
            )
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to submit claim: {e.response.text}")
            return {
                "error": f"Failed to submit claim: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error submitting claim: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_claim_status(self, claim_id: str) -> Dict[str, Any]:
        """
        Get the status of an existing claim.

        Args:
            claim_id: Unique claim identifier

        Returns:
            Dict containing claim status and details
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/{claim_id}/status",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Retrieved claim status for {claim_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get claim status: {e.response.text}")
            return {
                "error": f"Failed to get claim status: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error getting claim status: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_claim_details(self, claim_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a claim.

        Args:
            claim_id: Unique claim identifier

        Returns:
            Dict containing full claim details
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/{claim_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Retrieved claim details for {claim_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get claim details: {e.response.text}")
            return {
                "error": f"Failed to get claim details: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error getting claim details: {e}", exc_info=True)
            return {"error": str(e)}

    async def update_claim(
        self,
        claim_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update an existing claim.

        Args:
            claim_id: Unique claim identifier
            updates: Dictionary of fields to update (e.g., {"status": "approved", "amount": 5000})

        Returns:
            Dict containing updated claim information
        """
        try:
            response = await self.client.patch(
                f"{self.base_url}/{claim_id}",
                json=updates,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Updated claim {claim_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to update claim: {e.response.text}")
            return {
                "error": f"Failed to update claim: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error updating claim: {e}", exc_info=True)
            return {"error": str(e)}

    async def list_claims(
        self,
        policy_number: Optional[str] = None,
        status: Optional[str] = None,
        claim_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List claims for the tenant, optionally filtered.

        Args:
            policy_number: Filter by policy number
            status: Filter by status (e.g., "pending", "approved", "rejected")
            claim_type: Filter by claim type
            limit: Maximum number of claims to return
            offset: Offset for pagination

        Returns:
            Dict containing list of claims and pagination info
        """
        try:
            params = {
                "limit": limit,
                "offset": offset,
            }
            if policy_number:
                params["policy_number"] = policy_number
            if status:
                params["status"] = status
            if claim_type:
                params["claim_type"] = claim_type

            response = await self.client.get(
                f"{self.base_url}",
                params=params,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Retrieved {len(result.get('claims', []))} claims for tenant {self.tenant_id}"
            )
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to list claims: {e.response.text}")
            return {
                "error": f"Failed to list claims: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error listing claims: {e}", exc_info=True)
            return {"error": str(e)}

    async def upload_claim_document(
        self,
        claim_id: str,
        document_url: str,
        document_type: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Attach a document to a claim.

        Args:
            claim_id: Unique claim identifier
            document_url: URL of the document (from document upload service)
            document_type: Type of document (e.g., "receipt", "photo", "medical_report", "police_report")
            description: Optional description of the document

        Returns:
            Dict containing document attachment confirmation
        """
        try:
            validate_outbound_url(
                document_url,
                allowed_hosts=self.allowed_outbound_hosts,
                allow_localhost=False,
                require_https=True,
            )
            payload = {
                "document_url": document_url,
                "document_type": document_type,
            }
            if description:
                payload["description"] = description

            response = await self.client.post(
                f"{self.base_url}/{claim_id}/documents",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Uploaded document to claim {claim_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to upload claim document: {e.response.text}")
            return {
                "error": f"Failed to upload claim document: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error uploading claim document: {e}", exc_info=True)
            return {"error": str(e)}

    async def calculate_claim_estimate(
        self,
        claim_type: str,
        incident_description: str,
        policy_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate an estimated claim amount based on incident details.

        Args:
            claim_type: Type of claim (e.g., "auto", "property", "health")
            incident_description: Description of the incident
            policy_details: Optional policy details (coverage limits, deductibles, etc.)

        Returns:
            Dict containing estimated amount and breakdown
        """
        try:
            payload = {
                "claim_type": claim_type,
                "incident_description": incident_description,
                "tenant_id": self.tenant_id,
            }
            if policy_details:
                payload["policy_details"] = policy_details

            response = await self.client.post(
                f"{self.base_url}/estimate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Calculated claim estimate for {claim_type}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to calculate claim estimate: {e.response.text}")
            return {
                "error": f"Failed to calculate estimate: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error calculating claim estimate: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_claim_timeline(self, claim_id: str) -> Dict[str, Any]:
        """
        Get the timeline/status history of a claim.

        Args:
            claim_id: Unique claim identifier

        Returns:
            Dict containing timeline events
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/{claim_id}/timeline",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Retrieved claim timeline for {claim_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get claim timeline: {e.response.text}")
            return {
                "error": f"Failed to get claim timeline: {e.response.status_code}",
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"Error getting claim timeline: {e}", exc_info=True)
            return {"error": str(e)}


# Factory function to get claims tools instance
_claims_tools_instances: Dict[str, ClaimsTools] = {}


def get_claims_tools(
    tenant_id: str,
    backend_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    *,
    allowed_outbound_hosts: Optional[list] = None,
) -> ClaimsTools:
    """
    Factory function to get or create a ClaimsTools instance for a tenant.

    Args:
        tenant_id: Unique tenant identifier
        backend_url: Optional backend URL override
        auth_token: Optional JWT for the claims API
        allowed_outbound_hosts: Optional host allowlist for SSRF protection

    Returns:
        ClaimsTools instance
    """
    # Hash the token so raw secrets are never used as dict keys.
    cache_key = f"{tenant_id}:{hash_secret(auth_token)}"
    if cache_key not in _claims_tools_instances:
        _claims_tools_instances[cache_key] = ClaimsTools(
            tenant_id,
            backend_url,
            auth_token,
            allowed_outbound_hosts=allowed_outbound_hosts,
        )
    else:
        # Update auth token if it changed
        _claims_tools_instances[cache_key].auth_token = auth_token
    return _claims_tools_instances[cache_key]
