from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from domain.enums import ExecutionBackend
from domain.manifests import ImplementationMapping, NodeManifest

CompatibilityReasonCode = Literal[
    "missing_implementation",
    "unsupported_implementation",
    "unverified_implementation",
    "backend_excluded",
]


###############################################################################
@dataclass(frozen=True)
class ImplementationCompatibility:
    available: bool
    reason_code: CompatibilityReasonCode | None
    reason: str | None
    mapping: ImplementationMapping | None = None


###############################################################################
def resolve_implementation_compatibility(
    manifest: NodeManifest | None,
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> ImplementationCompatibility:
    """Resolve the exact verified node mapping for one static execution profile."""

    implementation_key = "keras" if keras_enabled else execution_backend.value
    implementation_label = (
        f"Keras on the {execution_backend.value} backend"
        if keras_enabled
        else f"native {execution_backend.value}"
    )
    display_name = manifest.display_name if manifest is not None else "This node"
    mapping = manifest.implementations.get(implementation_key) if manifest is not None else None
    if mapping is None:
        return ImplementationCompatibility(
            available=False,
            reason_code="missing_implementation",
            reason=f"{display_name} has no explicit {implementation_label} adapter.",
        )
    if not mapping.supported or mapping.status == "unsupported":
        return ImplementationCompatibility(
            available=False,
            reason_code="unsupported_implementation",
            reason=f"{display_name} is not supported by {implementation_label}.",
            mapping=mapping,
        )
    if mapping.status != "verified":
        return ImplementationCompatibility(
            available=False,
            reason_code="unverified_implementation",
            reason=f"{display_name} has not been verified for {implementation_label}.",
            mapping=mapping,
        )
    if keras_enabled and mapping.execution_backends.get(execution_backend, True) is False:
        return ImplementationCompatibility(
            available=False,
            reason_code="backend_excluded",
            reason=f"{display_name} is not available through Keras on the {execution_backend.value} backend.",
            mapping=mapping,
        )
    return ImplementationCompatibility(
        available=True,
        reason_code=None,
        reason=None,
        mapping=mapping,
    )
