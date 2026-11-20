import numpy as np
import pandas as pd

from src.drift import check_drift, psi_numeric


def _df(rng, age_mu, n=3000):
    return pd.DataFrame({"age": rng.normal(age_mu, 15, n), "creatinine": rng.normal(1.1, 0.3, n),
                         "diagnosis_group": rng.choice(["a", "b"], n)})


def test_no_drift_on_same_distribution(trained):
    rng = np.random.default_rng(0)
    rep = check_drift(_df(rng, 50), _df(rng, 50), log=False)
    assert not rep["drifted"].any()


def test_drift_detected_and_logged(trained, tmp_path, caplog):
    rng = np.random.default_rng(0)
    import logging
    logging.getLogger("drift").propagate = True
    with caplog.at_level(logging.WARNING, logger="drift"):
        rep = check_drift(_df(rng, 50), _df(rng, 70))
    assert bool(rep.set_index("feature").loc["age", "drifted"])
    assert any("DRIFT detected" in m for m in caplog.messages)


def test_psi_is_zero_for_identical():
    s = pd.Series(np.random.default_rng(1).normal(size=2000))
    assert psi_numeric(s, s) < 1e-6
