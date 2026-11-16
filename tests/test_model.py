import torch

from synax.config import ModelConfig
from synax.model.gpt import GPT


def make_model() -> GPT:
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        dropout=0.0,
    )
    return GPT(cfg)


def test_forward_shapes_and_loss() -> None:
    model = make_model()
    x = torch.randint(0, 64, (3, 16))
    y = torch.randint(0, 64, (3, 16))

    logits, loss, presents = model(x, y)

    assert logits.shape == (3, 16, 64)
    assert loss is not None
    assert torch.isfinite(loss)
    assert presents is None
    assert model.num_parameters() > 0


def test_backward_populates_gradients() -> None:
    model = make_model()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))

    _, loss, _ = model(x, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_causality() -> None:
    model = make_model().eval()
    x = torch.randint(0, 64, (1, 12))
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 64

    with torch.no_grad():
        logits_a, _, _ = model(x)
        logits_b, _, _ = model(changed)

    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_kv_cache_matches_full_forward() -> None:
    model = make_model().eval()
    x = torch.randint(0, 64, (1, 10))

    with torch.no_grad():
        full_logits, _, _ = model(x)
        _, _, cache = model(x[:, :-1], use_cache=True)
        last_logits, _, _ = model(
            x[:, -1:],
            past_kv=cache,
            pos_offset=x.shape[1] - 1,
            use_cache=True,
        )

    assert torch.allclose(full_logits[:, -1], last_logits[:, -1], atol=1e-4)


def test_generate_length() -> None:
    model = make_model().eval()
    prompt = torch.tensor([[1, 2, 3]])
    output = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10)
    assert output.shape == (1, 8)
    assert output[0, :3].tolist() == [1, 2, 3]


def test_generate_greedy_deterministic() -> None:
    model = make_model().eval()
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    second = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert torch.equal(first, second)
