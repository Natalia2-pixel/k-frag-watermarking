import pytest
import torch

from kfrag.models import MessageProjector


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_message_projector_multiscale_shapes(batch_size: int) -> None:
    model = MessageProjector(message_channels=8)
    features = model(torch.randint(0, 2, (batch_size, 44, 4, 4)).float())
    assert set(features) == {4, 8, 16, 32, 64}
    for scale, feature in features.items():
        assert feature.shape == (batch_size, 8, scale, scale)
        assert torch.isfinite(feature).all()


def test_message_projector_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        MessageProjector()(torch.zeros(1, 44, 5, 4))
