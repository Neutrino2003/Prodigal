"""Test personas — data matches assignment sample accounts."""

from dataclasses import dataclass, field


@dataclass
class Persona:
    name: str
    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: float
    # Optional override script; populated per subclass
    _script: list[str] = field(default_factory=list, repr=False)

    _VALID_CARD = "4532015112830366"
    _VALID_CVV = "123"
    _VALID_EXPIRY = "12/2027"

    def get_conversation_script(self) -> list[str]:
        if self._script:
            return list(self._script)
        return [
            self.account_id,
            self.full_name,
            self.dob,
            "500",
            f"{self._VALID_CARD} {self._VALID_CVV} {self._VALID_EXPIRY} {self.full_name}",
        ]

    # ── convenience ──────────────────────────────────────────────────────────
    def valid_card(self) -> str:
        return self._VALID_CARD

    def valid_cvv(self) -> str:
        return self._VALID_CVV


# ── Primary personas (match assignment sample accounts) ───────────────────────
PERSONAS: dict[str, "Persona"] = {
    "ACC1001": Persona(
        name="ACC1001",
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
    "ACC1002": Persona(
        name="ACC1002",
        account_id="ACC1002",
        full_name="Rajarajeswari Balasubramaniam",
        dob="1985-11-23",
        aadhaar_last4="9876",
        pincode="400002",
        balance=540.0,
    ),
    "ACC1003": Persona(
        name="ACC1003",
        account_id="ACC1003",
        full_name="Priya Agarwal",
        dob="1992-08-10",
        aadhaar_last4="2468",
        pincode="400003",
        balance=0.0,
    ),
    "ACC1004": Persona(
        name="ACC1004",
        account_id="ACC1004",
        full_name="Rahul Mehta",
        dob="1988-02-29",          # leap-year edge case
        aadhaar_last4="1357",
        pincode="400004",
        balance=3200.5,
    ),
}


# ── Extended scenario personas ────────────────────────────────────────────────

@dataclass
class _FailedVerificationPersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return [
            self.account_id,
            self.full_name,
            "1990-01-01",           # wrong DOB
            self.full_name,
            "1990-01-02",           # wrong again
            self.full_name,
            "1990-01-03",           # exhausts 3 attempts
        ]


@dataclass
class _WrongAccountPersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return ["INVALID999"]


@dataclass
class _ZeroBalancePersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return [
            self.account_id,
            self.full_name,
            self.dob,
        ]


@dataclass
class _InvalidCardPersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return [
            self.account_id,
            self.full_name,
            self.dob,
            "500",
            "1234567890123456 123 12/2027 Test User",  # fails Luhn
            "4532015112830366 123 12/2027 Nithin Jain",
        ]


@dataclass
class _ExpiredCardPersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return [
            self.account_id,
            self.full_name,
            self.dob,
            "500",
            "4532015112830366 123 01/2020 Nithin Jain",  # expired
        ]


@dataclass
class _InsufficientBalancePersona(Persona):
    def get_conversation_script(self) -> list[str]:
        return [
            self.account_id,
            self.full_name,
            self.dob,
            "99999",          # exceeds balance
            "500",            # corrected amount
            f"{self._VALID_CARD} {self._VALID_CVV} {self._VALID_EXPIRY} {self.full_name}",
        ]


EXTENDED_PERSONAS: dict[str, Persona] = {
    "happy_path_acc1001": Persona(
        name="Happy Path - ACC1001",
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
    "failed_verification_acc1002": _FailedVerificationPersona(
        name="Failed Verification - ACC1002",
        account_id="ACC1002",
        full_name="Rajarajeswari Balasubramaniam",
        dob="1985-11-23",
        aadhaar_last4="9876",
        pincode="400002",
        balance=540.0,
    ),
    "wrong_account": _WrongAccountPersona(
        name="Wrong Account ID",
        account_id="INVALID999",
        full_name="Test User",
        dob="2000-01-01",
        aadhaar_last4="0000",
        pincode="000000",
        balance=0.0,
    ),
    "zero_balance_acc1003": _ZeroBalancePersona(
        name="Zero Balance - ACC1003",
        account_id="ACC1003",
        full_name="Priya Agarwal",
        dob="1992-08-10",
        aadhaar_last4="2468",
        pincode="400003",
        balance=0.0,
    ),
    "leap_year_dob_acc1004": Persona(
        name="Leap Year DOB - ACC1004",
        account_id="ACC1004",
        full_name="Rahul Mehta",
        dob="1988-02-29",
        aadhaar_last4="1357",
        pincode="400004",
        balance=3200.5,
    ),
    "invalid_card_acc1001": _InvalidCardPersona(
        name="Invalid Card - ACC1001",
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
    "expired_card_acc1001": _ExpiredCardPersona(
        name="Expired Card - ACC1001",
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
    "insufficient_balance_acc1001": _InsufficientBalancePersona(
        name="Insufficient Balance - ACC1001",
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
}

ALL_PERSONAS: dict[str, Persona] = {**PERSONAS, **EXTENDED_PERSONAS}
