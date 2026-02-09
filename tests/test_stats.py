from capitalmarket.capitalselector.stats import EWMAStats


def test_ewma_converges_on_constant_pi():
    stats = EWMAStats(beta=0.1, mu=0.0, var=1.0)
    for _ in range(200):
        stats.update(1.0)
    assert abs(stats.mu - 1.0) < 1e-3
    assert stats.var < 1e-3


def test_drawdown_zero_on_positive_sequence():
    stats = EWMAStats(beta=0.1)
    for _ in range(10):
        stats.update(1.0)
        assert stats.dd == 0.0


def test_drawdown_increases_after_peak():
    stats = EWMAStats(beta=0.1)
    for _ in range(5):
        stats.update(1.0)
    dds = []
    for _ in range(5):
        stats.update(-0.5)
        dds.append(stats.dd)
    assert all(dds[i] <= dds[i + 1] for i in range(len(dds) - 1))
