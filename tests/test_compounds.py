import numpy as np
import pandas as pd

from analysis.build_compounds import nomination
from undercut import context


def test_nomination_ignores_citations_between_sentences():
    text = ("Tyre supplier [[Pirelli]] brought the C3, C4, and C5 tyre compounds for the event.<ref>{{cite}}</ref>"
            "Practice tested the C4, C5, and C6 compounds for next year.")
    nums, sentence = nomination(text)
    assert nums == [3, 4, 5] and sentence.startswith("Tyre supplier Pirelli brought")


def test_nomination_returns_none_without_a_nomination():
    assert nomination("Verstappen won the race from pole.")[0] is None


COMP = pd.DataFrame({"race_id": ["r"], "hard": [1], "medium": [3], "soft": [4]})


def test_hardness_maps_labels_to_that_races_c_numbers():
    d = pd.DataFrame({"race_id": ["r"], "chaser_old": ["SOFT"], "chaser_new": ["HARD"], "leader_old": ["MEDIUM"]})
    h = context.add_hardness(d, COMP).iloc[0]
    assert (h.old_c, h.new_c, h.leader_old_c) == (4, 1, 3)
    assert h.step_c == -3          # three steps harder (the race skipped C2)
    assert h.race_soft_c == 4


def test_hardness_is_nan_for_unknown_labels():
    d = pd.DataFrame({"race_id": ["r"], "chaser_old": ["UNKNOWN"], "chaser_new": ["HARD"], "leader_old": ["SOFT"]})
    h = context.add_hardness(d, COMP).iloc[0]
    assert np.isnan(h.old_c) and np.isnan(h.step_c) and h.new_c == 1


def test_compound_table_agrees_with_every_second_source():
    c = pd.read_csv("reference/compounds.csv").set_index("race_id")
    k = pd.read_csv("reference/compounds_checks.csv").set_index("race_id")
    assert len(c) == 87 and c[["hard", "medium", "soft"]].notna().all().all()
    assert (c.loc[k.index, ["hard", "medium", "soft"]].to_numpy() == k[["hard", "medium", "soft"]].to_numpy()).all()
