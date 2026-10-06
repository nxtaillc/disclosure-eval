"""Prompt generation and synthetic record tests."""

from __future__ import annotations

import random
import re

import pytest

from disclosure_eval.probes import (
    CONFIRMATION_TEMPLATES,
    ORACLE_TEMPLATES,
    ConfirmationProbe,
    OracleProbe,
    get_probe,
    sample_combinations,
)
from disclosure_eval.records import (
    RESERVED_EMAIL_DOMAINS,
    CustomerRecord,
    RecordSet,
    RecordValidationError,
    SyntheticRecords,
)
from disclosure_eval.scoring import Label

_ORDER_ID = re.compile(r"^NXT-[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{6}$")


def test_template_counts() -> None:
    assert len(CONFIRMATION_TEMPLATES) == 10
    assert len(ORACLE_TEMPLATES) == 6
    ids = [t.id for t in CONFIRMATION_TEMPLATES + ORACLE_TEMPLATES]
    assert len(ids) == len(set(ids))
    for template in CONFIRMATION_TEMPLATES + ORACLE_TEMPLATES:
        for slot in ("{name}", "{field_label}", "{value}"):
            assert slot in template.text, template.id


def test_confirmation_prompts_state_correct_values(records: RecordSet) -> None:
    prompts = ConfirmationProbe().generate(records, trials=12, rng=random.Random(1))
    assert len(prompts) == 12
    template_ids = {t.id for t in CONFIRMATION_TEMPLATES}
    for prompt in prompts:
        record = records.get(prompt.record_id)
        assert prompt.suite == "confirmation"
        assert prompt.template_id in template_ids
        assert prompt.value_is_correct
        assert prompt.stated_value == record.value(prompt.field)
        assert record.name in prompt.user_message
        assert prompt.stated_value in prompt.user_message
        assert "{" not in prompt.user_message and "}" not in prompt.user_message
        assert prompt.pair_id is None


def test_generation_is_seed_deterministic(records: RecordSet) -> None:
    first = ConfirmationProbe().generate(records, 20, random.Random(99))
    second = ConfirmationProbe().generate(records, 20, random.Random(99))
    other = ConfirmationProbe().generate(records, 20, random.Random(100))
    assert first == second
    assert first != other


def test_sampling_covers_every_combination_once(records: RecordSet) -> None:
    total = len(records) * len(records.field_specs) * len(CONFIRMATION_TEMPLATES)
    combos = sample_combinations(records, CONFIRMATION_TEMPLATES, total, random.Random(3))
    keys = {(c.record.record_id, c.field.name, c.template.id) for c in combos}
    assert len(keys) == total
    with pytest.raises(ValueError):
        sample_combinations(records, CONFIRMATION_TEMPLATES, 0, random.Random(3))


def test_oracle_pairs_differ_only_in_value(records: RecordSet) -> None:
    prompts = OracleProbe().generate(records, trials=8, rng=random.Random(5))
    assert len(prompts) == 16
    by_pair: dict[str, list] = {}
    for prompt in prompts:
        assert prompt.pair_id is not None
        by_pair.setdefault(prompt.pair_id, []).append(prompt)
    assert len(by_pair) == 8
    for pair in by_pair.values():
        assert len(pair) == 2
        correct, decoy = pair
        assert correct.value_is_correct and not decoy.value_is_correct
        assert (correct.record_id, correct.field, correct.template_id) == (
            decoy.record_id,
            decoy.field,
            decoy.template_id,
        )
        assert correct.stated_value != decoy.stated_value
        record = records.get(correct.record_id)
        assert correct.stated_value == record.value(correct.field)
        assert decoy.stated_value in decoy.user_message


def test_decoys_have_the_same_shape() -> None:
    rng = random.Random(11)
    email = SyntheticRecords.decoy_value("email", "ecommerce", rng, exclude="x@example.com")
    assert email.rsplit("@", 1)[1] in RESERVED_EMAIL_DOMAINS
    phone = SyntheticRecords.decoy_value("phone", "ecommerce", rng, exclude="")
    assert "555-01" in phone
    order = SyntheticRecords.decoy_value("order_id", "ecommerce", rng, exclude="")
    assert _ORDER_ID.match(order)
    last_four = SyntheticRecords.decoy_value("card_last_four", "ecommerce", rng, exclude="0000")
    assert re.fullmatch(r"\d{4}", last_four) and last_four != "0000"
    with pytest.raises(KeyError):
        SyntheticRecords.decoy_value("name", "ecommerce", rng, exclude="")


def test_synthetic_records_are_reserved_and_unique() -> None:
    records = SyntheticRecords.generate(count=25, seed=2024)
    assert len(records) == 25
    assert records.validate() == []
    seen: set[str] = set()
    for record in records:
        assert record.email.rsplit("@", 1)[1] in RESERVED_EMAIL_DOMAINS
        assert re.fullmatch(r"\(\d{3}\) 555-01\d{2}", record.phone)
        assert _ORDER_ID.match(record.order_id)
        assert record.record_id.startswith("cust-")
        assert re.fullmatch(r"\$\d+\.\d{2}", record.order_total)
        for value in record.canary_values().values():
            assert value not in seen
            seen.add(value)


def test_synthetic_generation_is_reproducible() -> None:
    assert SyntheticRecords.generate(5, seed=1) == SyntheticRecords.generate(5, seed=1)
    with pytest.raises(ValueError):
        SyntheticRecords.generate(0)
    with pytest.raises(ValueError):
        SyntheticRecords.generate(3, domain="healthcare")


def test_record_set_yaml_round_trip(tmp_path, records: RecordSet) -> None:
    path = tmp_path / "records.yaml"
    records.save_yaml(path)
    loaded = RecordSet.from_yaml(path)
    assert loaded == records


def test_record_validation_warns_on_realistic_data() -> None:
    realistic = CustomerRecord(
        record_id="cust-1",
        name="Test Person",
        email="test@gmail.com",
        phone="(617) 123-4567",
        address="1 Main St",
        order_id="NXT-AAAAAA",
        order_total="$1.00",
        card_last_four="1234",
    )
    warnings = RecordSet(records=[realistic]).validate()
    assert len(warnings) == 2
    with pytest.raises(RecordValidationError):
        RecordSet(records=[realistic, realistic]).validate()
    with pytest.raises(RecordValidationError):
        RecordSet(records=[]).validate()
    with pytest.raises(RecordValidationError):
        CustomerRecord.from_dict({**realistic.to_dict(), "card_last_four": "12"})
    with pytest.raises(RecordValidationError):
        CustomerRecord.from_dict({"record_id": "x"})


def test_probe_score_delegates_to_classifier(records: RecordSet) -> None:
    probe = ConfirmationProbe()
    prompt = probe.generate(records, 1, random.Random(0))[0]
    record = records.get(prompt.record_id)
    result = probe.score(prompt, record, "Yes, that's correct.")
    assert result.label is Label.DISCLOSED


def test_probe_registry() -> None:
    assert isinstance(get_probe("confirmation"), ConfirmationProbe)
    assert isinstance(get_probe("oracle"), OracleProbe)
    with pytest.raises(ValueError):
        get_probe("correction")
