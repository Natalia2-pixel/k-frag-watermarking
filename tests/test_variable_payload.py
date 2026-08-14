import torch
import pytest

from kfrag.training.variable_payload import (
    PairingTracker,
    generate_payload_splits,
    payload_splits_are_disjoint,
    should_early_stop,
    run_variable_payload,
)


def test_one_image_receives_different_payloads():
    tracker = PairingTracker(2, 2)
    tracker.update(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
    assert (tracker.counts[0] > 0).sum().item() >= 2
    assert tracker.assignments_vary()


def test_heldout_payloads_do_not_overlap_training_payloads():
    training, heldout = generate_payload_splits(12, 5, seed=2026)
    assert training.shape == (12, 44, 4, 4)
    assert heldout.shape == (5, 44, 4, 4)
    assert payload_splits_are_disjoint(training, heldout)


def test_training_accuracy_alone_cannot_trigger_early_stopping():
    excellent = {"non_index_accuracy": 1.0, "regional_packet_accuracy": 1.0}
    weak_heldout = {"non_index_accuracy": .5, "regional_packet_accuracy": 0.0}
    clean_negative = {"regional_packet_accuracy": 0.0}
    assert not should_early_stop(excellent, weak_heldout, clean_negative)


def test_early_stop_requires_clean_negative_control():
    excellent = {"non_index_accuracy": 1.0, "regional_packet_accuracy": 1.0}
    false_positive = {"regional_packet_accuracy": .02}
    assert not should_early_stop(excellent, excellent, false_positive)


def test_authentication_material_is_rejected_from_configuration():
    with pytest.raises(ValueError, match="must not be supplied"):
        run_variable_payload({"secret_key": "must-not-be-saved"})
