"""Synthetic customer records and their schemas."""

from .schemas import (
    DOMAIN_FIELDS,
    ECOMMERCE_FIELDS,
    RESERVED_EMAIL_DOMAINS,
    CustomerRecord,
    FieldSpec,
    RecordSet,
    RecordValidationError,
    Sensitivity,
    field_specs_for_domain,
)
from .synthetic import SyntheticRecords

__all__ = [
    "DOMAIN_FIELDS",
    "ECOMMERCE_FIELDS",
    "RESERVED_EMAIL_DOMAINS",
    "CustomerRecord",
    "FieldSpec",
    "RecordSet",
    "RecordValidationError",
    "Sensitivity",
    "SyntheticRecords",
    "field_specs_for_domain",
]
