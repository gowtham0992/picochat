"""Scale-planner and model-registry endpoint wrappers."""

from picochat.web import registry_plan, scale_plan_plan


def test_scale_plan_returns_recipe():
    plan = scale_plan_plan({"target_params": "100m"})
    assert plan["estimated_parameters"] > 50_000_000
    assert plan["n_layer"] >= 1 and plan["n_embd"] >= 1
    assert len(plan["markdown"]) > 100


def test_scale_plan_parses_human_counts_and_scales():
    small = scale_plan_plan({"target_params": "10m"})
    big = scale_plan_plan({"target_params": "300m"})
    assert big["estimated_parameters"] > small["estimated_parameters"]


def test_registry_empty(tmp_path):
    assert registry_plan(tmp_path) == {"entries": []}
