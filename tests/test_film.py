import pytest

torch = pytest.importorskip("torch")

from spg_lcda.models.film import FeatureFiLM


def test_film_starts_as_identity_and_receives_gradients() -> None:
    film = FeatureFiLM(text_dim=8, channels=4)
    features = torch.randn(2, 4, 5, 5, requires_grad=True)
    text = torch.randn(2, 8)

    output = film(features, text)
    assert torch.equal(output, features)

    output.square().mean().backward()
    assert film.projection.weight.grad is not None
    assert torch.count_nonzero(film.projection.weight.grad) > 0
