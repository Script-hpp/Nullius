from repro_agent.harness.verify import agent_fault_suspected

ETT = {"repeat": {"mse": 1.2944, "mae": 0.7132}, "zero": {"mse": 1.1099, "mae": 0.7960}}
# Echte Wertepaare (Seed 0) der Experiment-Codes, im Harness gemessen
A1 = {"mse": 1.3944, "mae": 0.9845}     # schlechter als jede Baseline
B2 = {"mse": 1.0436, "mae": 0.7728}     # MSE unter Null, MAE ueber Repeat
A6 = {"mse": 0.8092, "mae": 0.6766}     # schwach, aber besser als alle Baselines
SMOKE = {"mse": 1.1606, "mae": 0.8599}  # kaputter Code aus dem Probelauf


def flag(m, target=0.375):
    return agent_fault_suspected("mse", m["mse"], target, ETT, m)


def test_real_pairs():
    assert flag(A1)
    assert flag(SMOKE)
    assert flag(B2)                # nur ueber den MAE
    assert flag(A6) is None


def test_b2_is_missed_when_only_claim_metric_is_checked():
    assert agent_fault_suspected("mse", B2["mse"], 0.375, ETT) is None
    assert "MAE" in flag(B2)


def test_repeat_only_baseline_would_miss_smoke_mse_but_all_baselines_catch():
    assert agent_fault_suspected("mse", 1.1606, 0.375, {"repeat": ETT["repeat"]}) is None
    assert agent_fault_suspected("mse", 1.1606, 0.375, ETT)


def test_paper_that_does_not_beat_baseline_is_not_blamed_on_agent():
    assert agent_fault_suspected("mse", 1.3, 1.25, ETT, {"mse": 1.3, "mae": 0.9}) is None


def test_accuracy_majority_baseline():
    base = {"majority": {"accuracy": 0.316}}
    assert agent_fault_suspected("accuracy", 0.10, 0.815, base)
    assert agent_fault_suspected("accuracy", 0.80, 0.815, base) is None


def test_no_baselines_no_flag():
    assert agent_fault_suspected("mse", 9.0, 0.375, None) is None
