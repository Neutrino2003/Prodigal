from dataclasses import dataclass
import logging
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import settings

logger = logging.getLogger(__name__)


class AccountData(BaseModel):
    """Account data retrieved from the lookup API."""
    model_config = ConfigDict(extra="forbid")

    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: float



@dataclass
class LookupSuccess:
    account_data: AccountData


@dataclass
class LookupNotFound:
    pass


@dataclass
class LookupError:
    message: str


LookupResult = LookupSuccess | LookupNotFound | LookupError


@dataclass
class PaymentSuccess:
    transaction_id: str


@dataclass
class PaymentFailure:
    error_code: str


@dataclass
class PaymentError:
    message: str


PaymentResult = PaymentSuccess | PaymentFailure | PaymentError


class CardDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cardholder_name: str
    card_number: str
    cvv: str
    expiry_month: int
    expiry_year: int


class PaymentMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["card"]
    card: CardDetails


class PaymentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str
    amount: float
    payment_method: PaymentMethod


def lookup_account(account_id: str) -> LookupResult:
    url = f"{settings.payment_api_base_url}/api/lookup-account"
    payload = {"account_id": account_id}

    for attempt in range(settings.lookup_network_retries + 1):
        try:
            logger.info("lookup_account request attempt=%d account_id=%s", attempt + 1, account_id)
            with httpx.Client(timeout=settings.request_timeout_seconds) as client:
                response = client.post(url, json=payload)
            break
        except httpx.RequestError as exc:
            logger.warning("lookup_account network_error attempt=%d detail=%s", attempt + 1, str(exc)[:120])
            if attempt == settings.lookup_network_retries:
                return LookupError(f"Lookup request failed: {str(exc)[:120]}")
    else:
        return LookupError("Lookup request failed")

    if response.status_code == 404:
        logger.info("lookup_account not_found account_id=%s", account_id)
        return LookupNotFound()
    if response.status_code != 200:
        logger.error("lookup_account api_error status=%d", response.status_code)
        return LookupError(f"Lookup API error: {response.status_code}")

    try:
        account_data = AccountData(**response.json())
    except ValidationError as exc:
        logger.error("lookup_account invalid_response detail=%s", str(exc)[:200])
        return LookupError(f"Invalid lookup response: {str(exc)[:200]}")
    logger.info("lookup_account success account_id=%s", account_id)
    return LookupSuccess(account_data)


def process_payment(payload: PaymentPayload | dict) -> PaymentResult:
    try:
        parsed_payload = PaymentPayload.model_validate(payload)
    except ValidationError as exc:
        logger.error("process_payment invalid_payload detail=%s", str(exc)[:200])
        return PaymentError(f"Invalid payment payload: {str(exc)[:200]}")

    url = f"{settings.payment_api_base_url}/api/process-payment"
    logger.info(
        "process_payment request account_id=%s amount=%.2f",
        parsed_payload.account_id,
        parsed_payload.amount,
    )
    try:
        with httpx.Client(timeout=settings.payment_timeout_seconds) as client:
            response = client.post(url, json=parsed_payload.model_dump())
    except httpx.RequestError as exc:
        logger.error("process_payment network_error detail=%s", str(exc)[:120])
        return PaymentError(f"Payment request failed: {str(exc)[:120]}")

    if response.status_code == 200:
        transaction_id = response.json().get("transaction_id")
        if not transaction_id:
            logger.error("process_payment missing_transaction_id")
            return PaymentError("Missing transaction_id in payment response")
        logger.info("process_payment success transaction_id=%s", transaction_id)
        return PaymentSuccess(transaction_id=transaction_id)

    if response.status_code in {400, 422}:
        error_code = response.json().get("error_code", "unknown")
        logger.warning("process_payment failure status=%d error_code=%s", response.status_code, error_code)
        return PaymentFailure(error_code=error_code)

    logger.error("process_payment api_error status=%d", response.status_code)
    return PaymentError(f"Payment API error: {response.status_code}")
