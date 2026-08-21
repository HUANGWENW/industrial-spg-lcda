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


def test_evaluation_prompt_is_expanded_to_batch() -> None:
    from spg_lcda.models.film import P5FiLMHook

    film = P5FiLMHook(FeatureFiLM(text_dim=8, channels=4))
    film.set_evaluation_text_feature(torch.randn(8))
    source_module = torch.nn.Identity().eval()
    features = torch.randn(3, 4, 5, 5)
    output = film.apply(source_module, (features,), features)
    assert torch.equal(output, features)


def test_evaluation_accepts_one_prompt_per_image() -> None:
    from spg_lcda.models.film import P5FiLMHook

    film = P5FiLMHook(FeatureFiLM(text_dim=8, channels=4))
    film.set_evaluation_text_features(torch.randn(3, 8))
    source_module = torch.nn.Identity().eval()
    features = torch.randn(3, 4, 5, 5)
    output = film.apply(source_module, (features,), features)
    assert torch.equal(output, features)


def test_shift_alignment_is_zero_for_matching_directions() -> None:
    from spg_lcda.models.shift import ShiftAlignmentHead

    head = ShiftAlignmentHead(channels=2, text_dim=2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.eye(2))
    features = torch.tensor([[[[0.0]], [[0.0]]], [[[1.0]], [[0.0]]]])
    text = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    head.set_batch(text, torch.tensor([[0, 1]]))

    loss, cosine, pairs = head(features)

    assert torch.isclose(loss, torch.tensor(0.0))
    assert torch.isclose(cosine, torch.tensor(1.0))
    assert pairs.item() == 1
