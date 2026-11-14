"""Anti-Drone Sensor Log Analyzer (ADLA).

Parses RF detection logs, radar track logs, raw IQ/ADC dumps and equipment
error logs produced by a counter-UAS site, finds glitch / anomaly intervals,
computes FFT spectra, fuses RF and radar detections and writes a
self-contained HTML + CSV report.
"""

__version__ = "1.0.0"
