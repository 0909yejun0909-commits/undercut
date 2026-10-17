import numpy as np
import pandas as pd
import pytest

from analysis import rq1
from undercut import models


def test_mechanism_table_averages_all_parts_over_the_same_rows():
    d = pd.DataFrame({"success": [1.0, 1.0, 0.0], "mech_swing": [6.0, 77.0, -1.0], "mech_stop_diff": [2.0, np.nan, -1.0],
                      "mech_tyre_phase": [4.0, np.nan, 0.0], "mech_traffic": [0, 1, 0], "gap_s": [1.0, 1.0, 2.0],
                      "gain_s": [1.0, 1.0, 1.0], "margin_s": [1.0, 1.0, 0.0]})
    t = rq1.mechanism_table(d).set_index("success")
    assert t.at[1.0, "n"] == 1 and t.at[1.0, "mech_swing"] == pytest.approx(6.0)
    assert t.at[1.0, "mech_swing"] == pytest.approx(t.at[1.0, "mech_stop_diff"] + t.at[1.0, "mech_tyre_phase"])
    assert t.at[1.0, "median_mech_stop_diff"] == pytest.approx(2.0)


def test_lookup_holds_compound_at_the_most_common_choice():
    from tests.test_models import SIM
    d = models.prepare(SIM.assign(compound_change=["softer"] * 250 + ["same"] * 100 + ["harder"] * 50))
    res = models.fit_logit(d)
    assert set(rq1.lookup(res, d).compound_change) == {"softer"}
