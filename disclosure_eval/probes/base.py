"""Probe abstractions and prompt templates.

A probe turns a record set into a list of :class:`ProbePrompt` objects and
knows how to score the reply to each one. Prompts are rendered from plain
templates that read like ordinary customer messages; there are no
adversarial payloads. What makes a probe effective is that the values are
drawn from the developer's own records, so a confirmation is meaningful.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..records.schemas import CustomerRecord, FieldSpec, RecordSet
from ..scoring.classifier import Classification, classify


@dataclass(frozen=True)
class PromptTemplate:
    """A customer-message template with ``{name}``, ``{field_label}``, ``{value}`` slots."""

    id: str
    text: str

    def render(self, name: str, field_label: str, value: str) -> str:
        """Fill the template slots."""
        return self.text.format(name=name, field_label=field_label, value=value)


@dataclass(frozen=True)
class ProbePrompt:
    """A fully rendered probe ready to send to a target.

    Attributes:
        suite: Name of the probe that produced it.
        template_id: Identifier of the template used.
        record_id: Record the prompt claims to belong to.
        field: Record field the prompt asks about.
        stated_value: Value the prompt asserts for that field.
        value_is_correct: Whether ``stated_value`` matches the record.
        user_message: The rendered text sent as the user turn.
        pair_id: Shared identifier for oracle correct/incorrect pairs.
    """

    suite: str
    template_id: str
    record_id: str
    field: str
    stated_value: str
    value_is_correct: bool
    user_message: str
    pair_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "suite": self.suite,
            "template_id": self.template_id,
            "record_id": self.record_id,
            "field": self.field,
            "stated_value": self.stated_value,
            "value_is_correct": self.value_is_correct,
            "user_message": self.user_message,
            "pair_id": self.pair_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbePrompt:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            suite=str(data["suite"]),
            template_id=str(data["template_id"]),
            record_id=str(data["record_id"]),
            field=str(data["field"]),
            stated_value=str(data["stated_value"]),
            value_is_correct=bool(data["value_is_correct"]),
            user_message=str(data["user_message"]),
            pair_id=data.get("pair_id"),
        )


@dataclass(frozen=True)
class Combination:
    """One (record, field, template) choice a probe can render."""

    record: CustomerRecord
    field: FieldSpec
    template: PromptTemplate


def sample_combinations(
    records: RecordSet,
    templates: Sequence[PromptTemplate],
    count: int,
    rng: random.Random,
) -> list[Combination]:
    """Pick ``count`` (record, field, template) combinations.

    The full product is shuffled with ``rng`` so a small run still spreads
    across records, fields, and templates. If ``count`` exceeds the number
    of distinct combinations, the shuffled list is cycled.

    Raises:
        ValueError: If ``count`` is not positive or there is nothing to sample.
    """
    if count <= 0:
        raise ValueError("count must be a positive integer")
    product = [
        Combination(record=record, field=spec, template=template)
        for record in records
        for spec in records.field_specs
        for template in templates
    ]
    if not product:
        raise ValueError("no records, fields, or templates to sample from")
    rng.shuffle(product)
    return [product[i % len(product)] for i in range(count)]


class Probe(ABC):
    """Base class for probe suites."""

    name: ClassVar[str] = "abstract"
    templates: ClassVar[tuple[PromptTemplate, ...]] = ()

    @abstractmethod
    def generate(self, records: RecordSet, trials: int, rng: random.Random) -> list[ProbePrompt]:
        """Render ``trials`` worth of prompts from ``records``.

        The meaning of ``trials`` is probe-specific: the confirmation probe
        treats it as a prompt count, the oracle probe as a pair count.
        """

    def score(
        self,
        prompt: ProbePrompt,
        record: CustomerRecord,
        response_text: str,
        stop_reason: str | None = None,
    ) -> Classification:
        """Classify a reply to ``prompt``; delegates to the shared classifier."""
        return classify(
            response_text=response_text,
            record=record,
            probed_field=prompt.field,
            stated_value=prompt.stated_value,
            value_is_correct=prompt.value_is_correct,
            stop_reason=stop_reason,
        )
