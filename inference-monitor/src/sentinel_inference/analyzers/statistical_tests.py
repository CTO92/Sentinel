"""Kolmogorov-Smirnov and Anderson-Darling tests against baseline distributions.

Maintains a rolling window of logit samples as a baseline and performs
two-sample KS and Anderson-Darling tests to detect distributional shifts
caused by silent data corruption.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import structlog
from scipy import stats as sp_stats

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import StatisticalTestsConfig

logger = structlog.get_logger(__name__)


class StatisticalTestAnalyzer:
    """Two-sample statistical tests comparing current vs. baseline logit distributions.

    Maintains a rolling baseline of flattened logit samples and runs
    KS and Anderson-Darling tests on each new sample.

    Parameters
    ----------
    config : StatisticalTestsConfig
        Test parameters and thresholds.
    """

    def __init__(self, config: StatisticalTestsConfig | None = None) -> None:
        self._config = config or StatisticalTestsConfig()
        self._baseline: deque[np.ndarray] = deque(maxlen=self._config.baseline_size)
        self._baseline_flat: np.ndarray | None = None
        self._needs_rebuild = True

    @property
    def baseline_size(self) -> int:
        return len(self._baseline)

    def _rebuild_baseline(self) -> None:
        """Concatenate baseline samples into a single array for testing."""
        if self._baseline:
            self._baseline_flat = np.concatenate([s.ravel() for s in self._baseline])
        else:
            self._baseline_flat = None
        self._needs_rebuild = False

    def analyze(self, logits: np.ndarray) -> list[AnomalyEvent]:
        """Run statistical tests on a new logit sample.

        Parameters
        ----------
        logits : np.ndarray
            Output logit tensor.

        Returns
        -------
        list of AnomalyEvent for tests that reject the null hypothesis
        (same distribution).
        """
        if not self._config.enabled:
            return []

        flat = logits.astype(np.float64).ravel()

        # Need a minimum baseline before testing
        min_baseline = max(50, self._config.baseline_size // 10)
        if len(self._baseline) < min_baseline:
            self._baseline.append(flat)
            self._needs_rebuild = True
            return []

        if self._needs_rebuild:
            self._rebuild_baseline()

        assert self._baseline_flat is not None
        anomalies: list[AnomalyEvent] = []

        # Subsample baseline if it's very large to keep tests fast
        baseline = self._baseline_flat
        if len(baseline) > 50000:
            rng = np.random.default_rng(seed=0)
            indices = rng.choice(len(baseline), size=50000, replace=False)
            baseline = baseline[indices]

        # Subsample current if very large
        current = flat
        if len(current) > 50000:
            rng = np.random.default_rng(seed=1)
            indices = rng.choice(len(current), size=50000, replace=False)
            current = current[indices]

        # --- Kolmogorov-Smirnov Test ---
        ks_stat, ks_p = sp_stats.ks_2samp(current, baseline)
        if ks_p < self._config.ks_p_value_threshold:
            anomalies.append(
                AnomalyEvent(
                    analyzer="statistical_tests",
                    stat_name="ks_test",
                    observed_value=ks_stat,
                    ewma_value=ks_p,
                    ucl=self._config.ks_p_value_threshold,
                    lcl=0.0,
                    sample_count=len(self._baseline),
                    severity=(
                        "critical" if ks_p < self._config.ks_p_value_threshold / 10 else "warning"
                    ),
                    details={"ks_statistic": ks_stat, "p_value": ks_p},
                )
            )
            logger.warning(
                "ks_test_rejected",
                ks_statistic=ks_stat,
                p_value=ks_p,
                threshold=self._config.ks_p_value_threshold,
            )

        # --- Anderson-Darling k-sample Test ---
        try:
            ad_result = sp_stats.anderson_ksamp([current, baseline])
            ad_stat = ad_result.statistic
            # ad_result.significance_level is the approximate p-value
            ad_p = ad_result.significance_level

            # Anderson-Darling critical values at various significance levels
            # scipy returns significance_level as the p-value approximation
            # We reject if p-value < threshold corresponding to chosen level
            sig_levels = [0.15, 0.10, 0.05, 0.025, 0.01]
            chosen_level = sig_levels[min(self._config.ad_significance_level, len(sig_levels) - 1)]

            if ad_p < chosen_level:
                anomalies.append(
                    AnomalyEvent(
                        analyzer="statistical_tests",
                        stat_name="anderson_darling",
                        observed_value=ad_stat,
                        ewma_value=ad_p,
                        ucl=chosen_level,
                        lcl=0.0,
                        sample_count=len(self._baseline),
                        severity="critical" if ad_p < chosen_level / 10 else "warning",
                        details={
                            "ad_statistic": ad_stat,
                            "p_value": ad_p,
                            "significance_level": chosen_level,
                        },
                    )
                )
                logger.warning(
                    "anderson_darling_rejected",
                    ad_statistic=ad_stat,
                    p_value=ad_p,
                    significance_level=chosen_level,
                )
        except Exception as exc:
            logger.debug("anderson_darling_failed", error=str(exc))

        # Add to baseline (only if not anomalous, to prevent poisoning)
        if not anomalies:
            self._baseline.append(flat)
            self._needs_rebuild = True

        return anomalies

    def reset(self) -> None:
        self._baseline.clear()
        self._baseline_flat = None
        self._needs_rebuild = True
