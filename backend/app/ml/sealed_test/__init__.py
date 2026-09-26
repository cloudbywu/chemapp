"""Executable sealed-test state machine for the v8 external-holder protocol.

Four stages: signed+anchored reservation -> atomic attempt-start ->
complete-denominator pre-Gold prediction commitment -> post-Gold
evaluation receipt.  All artifacts stay fail-closed until a genuinely
independent external holder executes the protocol; this package provides the
machinery and forbids every unsafe transition.
"""

from .crypto import CryptoError
from .ledger import LedgerError, Ledger, Denylist
from .protocol import SealedTestError, SealedTestStateMachine

__all__ = [
    "CryptoError",
    "LedgerError",
    "Ledger",
    "Denylist",
    "SealedTestError",
    "SealedTestStateMachine",
]
