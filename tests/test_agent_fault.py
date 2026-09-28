from repro_agent.harness.verify import agent_fault_suspected

ETT = {"repeat": {"mse": 1.2944, "mae": 0.7132}, "zero": {"mse": 1.1099, "mae": 0.7960}}


def test_bad_code_from_smoke_run_is_flagged_on_mae_and_mse():
    assert agent_fault_suspected("mae", 0.8599, 0.399, ETT)
    assert agent_fault_suspected("mse", 1.1606, 0.375, ETT)   # schlechter als Nullvorhersage


def test_repeat_only_would_miss_it_but_best_baseline_catches():
    only_repeat = {"repeat": ETT["repeat"]}
    assert agent_fault_suspected("mse", 1.1606, 0.375, only_repeat) is None
    assert agent_fault_suspected("mse", 1.1606, 0.375, ETT)


def test_weak_but_valid_model_not_flagged():
    assert agent_fault_suspected("mse", 0.858, 0.375, ETT) is None


def test_paper_that_does_not_beat_baseline_is_not_blamed_on_agent():
    assert agent_fault_suspected("mse", 1.3, 1.25, ETT) is None      # Paperwert selbst schlechter als Baseline


def test_accuracy_majority_baseline():
    assert agent_fault_suspected("accuracy", 0.10, 0.815, {"majority": {"accuracy": 0.316}})
    assert agent_fault_suspected("accuracy", 0.80, 0.815, {"majority": {"accuracy": 0.316}}) is None


def test_no_baselines_no_flag():
    assert agent_fault_suspected("mse", 9.0, 0.375, None) is None
