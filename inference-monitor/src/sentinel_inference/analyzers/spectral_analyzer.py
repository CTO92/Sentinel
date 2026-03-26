"""FFT-based spectral analyzer for detecting SDC bit-flip signatures.

Silent data corruption caused by bit-flips in GPU SRAM/HBM creates
characteristic high-frequency components in logit vectors.  This analyzer
computes the power spectral density (PSD) of each logit vector and
compares the energy distribution against a rolling baseline.

Detection logic:
1. Compute real FFT of the logit vector.
2. Compute one-sided power spectral density.
3. Partition spectrum into low-frequency (bottom 75%) and high-frequency
   (top 25%) bands.
4. If the ratio of high-freq energy to total energy exceeds a threshold,
   flag as anomalous.
5. Maintain a rolling baseline of PSD profiles and alert when the current
   PSD deviates significantly.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import structlog

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import SpectralAnalyzerConfig

logger = structlog.get_logger(__name__)


def compute_psd(logits: np.ndarray) -> np.ndarray:
    """Compute one-sided power spectral density of a logit vector.

    Parameters
    ----------
    logits : np.ndarray
        1-D logit vector (or flattened).

    Returns
    -------
    np.ndarray
        One-sided PSD array of length ``N // 2 + 1``.
    """
    x = logits.astype(np.float64).ravel()
    n = len(x)
    # Remove DC component (mean)
    x = x - np.mean(x)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    x_windowed = x * window
    # Real FFT
    fft_vals = np.fft.rfft(x_windowed)
    # Power spectral density (magnitude squared), normalized
    psd = np.abs(fft_vals) ** 2 / n
    # Double the non-DC, non-Nyquist bins for one-sided spectrum
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return psd


def high_frequency_energy_ratio(psd: np.ndarray, cutoff_fraction: float = 0.75) -> float:
    """Fraction of total energy in the high-frequency band.

    Parameters
    ----------
    psd : np.ndarray
        One-sided PSD array.
    cutoff_fraction : float
        Fraction of bins considered "low frequency" (default 0.75 means
        top 25% of bins are high-frequency).

    Returns
    -------
    float
        Ratio of high-frequency energy to total energy, in [0, 1].
    """
    total = float(np.sum(psd))
    if total < 1e-30:
        return 0.0
    cutoff_idx = int(len(psd) * cutoff_fraction)
    high_freq_energy = float(np.sum(psd[cutoff_idx:]))
    return high_freq_energy / total


class SpectralAnalyzer:
    """FFT-based analyzer detecting SDC bit-flip spectral signatures.

    Parameters
    ----------
    config : SpectralAnalyzerConfig
        Threshold and baseline window parameters.
    """

    def __init__(self, config: SpectralAnalyzerConfig | None = None) -> None:
        self._config = config or SpectralAnalyzerConfig()
        self._baseline_ratios: deque[float] = deque(maxlen=self._config.baseline_window)
        self._baseline_psds: deque[np.ndarray] = deque(maxlen=self._config.baseline_window)

    @property
    def baseline_size(self) -> int:
        return len(self._baseline_ratios)

    def analyze(self, logits: np.ndarray) -> list[AnomalyEvent]:
        """Analyze a logit vector for anomalous spectral content.

        Parameters
        ----------
        logits : np.ndarray
            Output logit tensor.

        Returns
        -------
        list of AnomalyEvent.
        """
        if not self._config.enabled:
            return []

        flat = logits.ravel()
        if len(flat) < 8:
            return []

        psd = compute_psd(flat)
        hf_ratio = high_frequency_energy_ratio(psd)

        anomalies: list[AnomalyEvent] = []

        # Absolute threshold check
        if hf_ratio > self._config.high_freq_ratio_threshold:
            anomalies.append(
                AnomalyEvent(
                    analyzer="spectral_analyzer",
                    stat_name="high_freq_ratio",
                    observed_value=hf_ratio,
                    ewma_value=hf_ratio,
                    ucl=self._config.high_freq_ratio_threshold,
                    lcl=0.0,
                    sample_count=len(self._baseline_ratios),
                    severity="warning",
                    details={"psd_len": len(psd)},
                )
            )
            logger.warning(
                "spectral_anomaly_hf_ratio",
                hf_ratio=hf_ratio,
                threshold=self._config.high_freq_ratio_threshold,
            )

        # Baseline comparison (after sufficient samples)
        if len(self._baseline_ratios) >= self._config.baseline_window // 2:
            baseline_arr = np.array(self._baseline_ratios)
            mean_ratio = float(np.mean(baseline_arr))
            std_ratio = float(np.std(baseline_arr))

            if std_ratio > 1e-12:
                z_score = (hf_ratio - mean_ratio) / std_ratio
                if abs(z_score) > 4.0:
                    anomalies.append(
                        AnomalyEvent(
                            analyzer="spectral_analyzer",
                            stat_name="high_freq_z_score",
                            observed_value=z_score,
                            ewma_value=mean_ratio,
                            ucl=mean_ratio + 4.0 * std_ratio,
                            lcl=mean_ratio - 4.0 * std_ratio,
                            sample_count=len(self._baseline_ratios),
                            severity="critical" if abs(z_score) > 6.0 else "warning",
                            details={
                                "baseline_mean": mean_ratio,
                                "baseline_std": std_ratio,
                            },
                        )
                    )

            # Cosine similarity of PSD profiles
            if self._baseline_psds and len(psd) == len(self._baseline_psds[-1]):
                baseline_psd = np.mean(
                    [p for p in self._baseline_psds if len(p) == len(psd)], axis=0
                )
                norm_current = np.linalg.norm(psd)
                norm_baseline = np.linalg.norm(baseline_psd)
                if norm_current > 1e-12 and norm_baseline > 1e-12:
                    cosine_sim = float(np.dot(psd, baseline_psd) / (norm_current * norm_baseline))
                    if cosine_sim < 0.8:
                        anomalies.append(
                            AnomalyEvent(
                                analyzer="spectral_analyzer",
                                stat_name="psd_cosine_similarity",
                                observed_value=cosine_sim,
                                ewma_value=1.0,
                                ucl=1.0,
                                lcl=0.8,
                                sample_count=len(self._baseline_psds),
                                severity="warning",
                            )
                        )

        # Update baseline (only with non-anomalous samples)
        if not anomalies:
            self._baseline_ratios.append(hf_ratio)
            self._baseline_psds.append(psd)

        return anomalies

    def reset(self) -> None:
        self._baseline_ratios.clear()
        self._baseline_psds.clear()
