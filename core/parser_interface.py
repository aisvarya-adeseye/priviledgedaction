from abc import ABC, abstractmethod
from typing import Any


class IntentParser(ABC):
    """
    Abstract interface for components that convert raw untrusted input
    into a tentative typed intent.

    A parser is not the final authority over privileged actions.
    It only proposes a tentative intent.
    """

    @abstractmethod
    def parse(
        self,
        trusted_state: dict[str, Any],
        raw_text: str,
        target: str | None = None,
        evidence: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """
        Parse raw text into a tentative typed intent.

        Returns:
            (tentative_intent, raw_parser_output)
        """
        raise NotImplementedError
