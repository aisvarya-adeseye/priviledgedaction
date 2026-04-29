"""Core package for the CPS project.

This file marks `core` as a regular package so tests and scripts
can import `core.*` reliably when the project root is on `sys.path`.
"""

__all__ = [
    "domain_adapter",
    "logging",
    "parser_interface",
    "pipeline",
    "types",
]
