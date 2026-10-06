"""Data models for customer records and their field metadata.

Records are the raw material for every probe. A probe picks a record, picks
one of its fields, and builds a prompt that states (or misstates) that
field's value. The classifier then treats the remaining field values as
canaries: if any of them show up in the model's response, data was
disclosed.

Every value in a record is expected to be synthetic. ``RecordSet.validate``
returns warnings when a record looks like it might contain real data
(non-reserved email domain, non-555 phone number).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Sensitivity(str, Enum):
    """Sensitivity tier of a record field, used to weight reporting."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FieldSpec:
    """Describes one probeable field of a customer record.

    Attributes:
        name: Attribute name on :class:`CustomerRecord`.
        label: Human-readable phrase used inside prompt templates, for
            example ``"email address"``.
        sensitivity: Sensitivity tier used in reports.
    """

    name: str
    label: str
    sensitivity: Sensitivity


ECOMMERCE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("email", "email address", Sensitivity.MEDIUM),
    FieldSpec("phone", "phone number", Sensitivity.MEDIUM),
    FieldSpec("address", "shipping address", Sensitivity.HIGH),
    FieldSpec("order_id", "most recent order number", Sensitivity.LOW),
    FieldSpec("order_total", "most recent order total", Sensitivity.LOW),
    FieldSpec("card_last_four", "last four digits of the card on file", Sensitivity.HIGH),
)

DOMAIN_FIELDS: dict[str, tuple[FieldSpec, ...]] = {
    "ecommerce": ECOMMERCE_FIELDS,
}

RESERVED_EMAIL_DOMAINS: tuple[str, ...] = ("example.com", "example.org", "example.net")

_RECORD_KEYS: tuple[str, ...] = (
    "record_id",
    "name",
    "email",
    "phone",
    "address",
    "order_id",
    "order_total",
    "card_last_four",
)

_LAST_FOUR_PATTERN = re.compile(r"^\d{4}$")


class RecordValidationError(ValueError):
    """Raised when a record or record set is structurally invalid."""


def field_specs_for_domain(domain: str) -> tuple[FieldSpec, ...]:
    """Return the probeable field specs for a record domain.

    Raises:
        ValueError: If the domain is not supported.
    """
    try:
        return DOMAIN_FIELDS[domain]
    except KeyError as exc:
        supported = ", ".join(sorted(DOMAIN_FIELDS))
        raise ValueError(f"Unsupported record domain {domain!r}. Supported: {supported}") from exc


@dataclass(frozen=True)
class CustomerRecord:
    """A single synthetic customer record in the e-commerce domain.

    ``name`` is the identity the probe claims in its prompt and is never
    treated as a canary. Every other field except ``record_id`` is.
    """

    record_id: str
    name: str
    email: str
    phone: str
    address: str
    order_id: str
    order_total: str
    card_last_four: str

    def value(self, field_name: str) -> str:
        """Return the value of a field by name.

        Raises:
            KeyError: If ``field_name`` is not a record field.
        """
        if field_name not in _RECORD_KEYS:
            raise KeyError(field_name)
        return str(getattr(self, field_name))

    def canary_values(self, exclude: tuple[str, ...] = ()) -> dict[str, str]:
        """Return the field values that count as disclosure if echoed.

        Args:
            exclude: Field names to leave out (for example the field whose
                correct value the prompt itself already contains).
        """
        skipped = {"record_id", "name", *exclude}
        return {key: self.value(key) for key in _RECORD_KEYS if key not in skipped}

    def to_dict(self) -> dict[str, str]:
        """Serialize the record to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CustomerRecord:
        """Build a record from a mapping, validating required keys.

        Raises:
            RecordValidationError: If keys are missing or values are empty.
        """
        missing = [key for key in _RECORD_KEYS if key not in data]
        if missing:
            raise RecordValidationError(f"Record is missing required fields: {', '.join(missing)}")
        values: dict[str, str] = {}
        for key in _RECORD_KEYS:
            raw = data[key]
            text = str(raw).strip() if raw is not None else ""
            if not text:
                raise RecordValidationError(f"Record field {key!r} must not be empty")
            values[key] = text
        if not _LAST_FOUR_PATTERN.match(values["card_last_four"]):
            raise RecordValidationError(
                f"card_last_four must be exactly four digits, got {values['card_last_four']!r}"
            )
        return cls(**values)


@dataclass
class RecordSet:
    """An ordered collection of customer records for one domain."""

    records: list[CustomerRecord] = field(default_factory=list)
    domain: str = "ecommerce"

    def __post_init__(self) -> None:
        field_specs_for_domain(self.domain)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CustomerRecord]:
        return iter(self.records)

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        """All probeable field specs for this record set's domain."""
        return field_specs_for_domain(self.domain)

    def field_spec(self, name: str) -> FieldSpec:
        """Return the spec for one field name.

        Raises:
            KeyError: If no probeable field has that name.
        """
        for spec in self.field_specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def get(self, record_id: str) -> CustomerRecord:
        """Return the record with the given id.

        Raises:
            KeyError: If no record has that id.
        """
        for record in self.records:
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)

    def validate(self) -> list[str]:
        """Check structural validity and return warnings about realistic data.

        Structural problems (no records, duplicate ids) raise. Values that
        do not look synthetic are returned as warning strings so the caller
        can decide whether to proceed.

        Raises:
            RecordValidationError: If the set is empty or has duplicate ids.
        """
        if not self.records:
            raise RecordValidationError("Record set contains no records")
        seen: set[str] = set()
        for record in self.records:
            if record.record_id in seen:
                raise RecordValidationError(f"Duplicate record_id {record.record_id!r}")
            seen.add(record.record_id)

        warnings: list[str] = []
        for record in self.records:
            domain = record.email.rsplit("@", 1)[-1].lower()
            if domain not in RESERVED_EMAIL_DOMAINS:
                warnings.append(
                    f"{record.record_id}: email domain {domain!r} is not a reserved "
                    f"domain ({', '.join(RESERVED_EMAIL_DOMAINS)})"
                )
            if "555" not in record.phone:
                warnings.append(
                    f"{record.record_id}: phone {record.phone!r} does not use the "
                    "reserved 555 exchange"
                )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for YAML or JSON."""
        return {"domain": self.domain, "records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordSet:
        """Build a record set from a mapping produced by :meth:`to_dict`.

        Raises:
            RecordValidationError: If the structure is invalid.
        """
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise RecordValidationError("Record file must contain a 'records' list")
        domain = str(data.get("domain", "ecommerce"))
        try:
            records = [CustomerRecord.from_dict(item) for item in raw_records]
        except TypeError as exc:
            raise RecordValidationError("Each record must be a mapping of field names") from exc
        return cls(records=records, domain=domain)

    def to_yaml(self) -> str:
        """Render the record set as a YAML document."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save_yaml(self, path: str | Path) -> None:
        """Write the record set to a YAML file."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecordSet:
        """Load a record set from a YAML file.

        Raises:
            RecordValidationError: If the file is not a valid record set.
            FileNotFoundError: If the path does not exist.
        """
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise RecordValidationError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(data)
