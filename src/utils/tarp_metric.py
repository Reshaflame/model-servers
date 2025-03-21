# src/utils/tarp_metric.py

import numpy as np

class TaPR:
    def __init__(self, delta=0.5):
        """
        delta: ratio for ambiguous regions (e.g., 0.5 means ambiguous region length = 50% of anomaly length)
        """
        self.delta = delta

    def _get_ambiguous_section(self, anomaly):
        start, end = anomaly
        length = end - start
        ambiguous_start = end + 1
        ambiguous_end = ambiguous_start + int(length * self.delta)
        return (ambiguous_start, ambiguous_end)

    def evaluate(self, anomalies, predictions):
        """
        anomalies: list of tuples [(start, end), ...] for ground truth anomalies
        predictions: list of tuples [(start, end), ...] for predicted anomaly ranges
        Returns: dict of TaP, TaR, TaPR
        """
        tp_d, tp_p = 0, 0  # Detection and Portion scores
        detected_anomalies = set()

        for anomaly in anomalies:
            start_gt, end_gt = anomaly
            ambiguous_start, ambiguous_end = self._get_ambiguous_section(anomaly)

            detected = False
            for pred_start, pred_end in predictions:
                if pred_end >= start_gt and pred_start <= end_gt:
                    detected = True
                    overlap = max(0, min(pred_end, end_gt) - max(pred_start, start_gt) + 1)
                    tp_p += overlap / (end_gt - start_gt + 1)  # Portion score
            if detected:
                detected_anomalies.add(anomaly)
                tp_d += 1

        n_anomalies = len(anomalies)
        n_preds = len(predictions)

        taP = (tp_d + tp_p) / (n_preds or 1)
        taR = (tp_d + tp_p) / (n_anomalies or 1)
        taPR = 2 * taP * taR / (taP + taR + 1e-8)  # F1-like combination

        return {"TaP": taP, "TaR": taR, "TaPR": taPR}

