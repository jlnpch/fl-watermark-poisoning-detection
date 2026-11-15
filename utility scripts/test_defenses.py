"""Unit tests for pytorchexample.defenses (NumPy-only, no Flower/torch)."""

import numpy as np

from pytorchexample.defenses import (
    run_defense,
)


def _models(*arrays):
    return [{"w": np.asarray(a, dtype=np.float32)} for a in arrays]


def test_fedavg():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    models = _models([1.0, 1.0], [3.0, 3.0])  # equal weights -> mean [2,2]
    out = run_defense("none", global_model, models, [1, 1])
    np.testing.assert_allclose(out.aggregated["w"], [2.0, 2.0])
    assert out.kept == [0, 1] and not any(out.excluded)

    # weighted: num_examples 3:1 -> 0.75*[1,1] + 0.25*[3,3] = [1.5,1.5]
    out = run_defense("none", global_model, models, [3, 1])
    np.testing.assert_allclose(out.aggregated["w"], [1.5, 1.5])


def test_mad_excludes_high_ber():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    # honest models near [2,2]; attacker far at [100,100]
    models = _models([2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [100.0, 100.0])
    ber = [0.01, 0.01, 0.02, 0.9]
    out = run_defense("mad", global_model, models, [1, 1, 1, 1], ber_values=ber, mad_alpha=4.0)
    assert out.excluded[3] is True
    assert 3 not in out.kept
    # aggregated should be near [2,2] (attacker excluded)
    np.testing.assert_allclose(out.aggregated["w"], [2.0, 2.0], atol=1e-4)
    assert out.extra["method"] == "mad"


def test_fixed_threshold():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    models = _models([1.0, 1.0], [1.0, 1.0], [50.0, 50.0])
    out = run_defense("fixed", global_model, models, [1, 1, 1], ber_values=[0.01, 0.02, 0.8], max_ber=0.3)
    assert out.excluded[2] is True
    np.testing.assert_allclose(out.aggregated["w"], [1.0, 1.0], atol=1e-4)


def test_krum_excludes_outlier():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    # 4 honest clustered updates + 1 far outlier. n=5, f=1 -> m=2, k=2.
    models = _models(
        [1.0, 1.0], [1.0, 1.1], [0.9, 1.0], [1.0, 0.9], [100.0, -100.0]
    )
    out = run_defense("krum", global_model, models, [1] * 5, n_attackers=1)
    assert out.extra["m"] == 2
    assert 4 not in out.kept  # the outlier should never be selected
    agg = out.aggregated["w"]
    assert abs(agg[0]) < 5 and abs(agg[1]) < 5


def test_trimmed_mean():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    models = _models([1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 100.0], [-100.0, -100.0])
    # trim 1 each side -> mean of [1,2,3] = 2
    out = run_defense("trimmed-mean", global_model, models, [1] * 5, trim_count=1)
    np.testing.assert_allclose(out.aggregated["w"], [2.0, 2.0], atol=1e-4)


def test_median():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    models = _models([1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 100.0], [-100.0, -100.0])
    out = run_defense("median", global_model, models, [1] * 5)
    np.testing.assert_allclose(out.aggregated["w"], [2.0, 2.0], atol=1e-4)


def test_fltrust():
    global_model = {"w": np.array([0.0, 0.0], dtype=np.float32)}
    server_update = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    # client 0 aligned with server, client 1 anti-aligned (cos < 0 -> ReLU 0)
    models = _models([2.0, 2.0], [-2.0, -2.0])
    out = run_defense("fltrust", global_model, models, [1, 1], server_update=server_update)
    assert out.extra["trust_scores"][0] > 0.99
    assert out.extra["trust_scores"][1] == 0.0
    # only client 0 trusted -> result direction positive
    assert out.aggregated["w"][0] > 0 and out.aggregated["w"][1] > 0


def test_unknown_method_raises():
    global_model = {"w": np.zeros(2, dtype=np.float32)}
    try:
        run_defense("bogus", global_model, [], [])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_strategy_call_shape():
    """Mimic the strategy's full dispatch (all cfg keys present, some ignored)."""
    global_model = {"w": np.zeros(2, dtype=np.float32)}
    models = _models([1.0, 1.0], [2.0, 2.0])
    full_cfg = dict(
        mad_alpha=4.0, max_ber=0.3, n_attackers=0, m=None,
        trim_count=0, server_update=None,
    )
    ber = [0.01, 0.02]
    for method in ["none", "mad", "fixed", "krum", "trimmed-mean", "median"]:
        out = run_defense(method, global_model, models, [1, 1], ber_values=ber, **full_cfg)
        assert out.aggregated is not None
    # fltrust with a real server update
    full_cfg["server_update"] = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    out = run_defense("fltrust", global_model, models, [1, 1], ber_values=ber, **full_cfg)
    assert out.aggregated is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
