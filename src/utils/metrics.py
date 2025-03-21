from sklearn.metrics import precision_score, recall_score, f1_score

class Metrics:
    def __init__(self):
        # from utils.tarp_metric import TaPR
        # self.tapr = TaPR()
        pass

    def compute_standard_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"Precision": precision, "Recall": recall, "F1": f1}

    # def compute_tapr(self, anomalies, predictions):
    #     return self.tapr.evaluate(anomalies, predictions)

    def compute_all(self, y_true, y_pred, anomaly_ranges=None, pred_ranges=None):
        standard = self.compute_standard_metrics(y_true, y_pred)
        # tapr = self.compute_tapr(anomaly_ranges, pred_ranges)
        # return {**standard, **tapr}
        return {**standard}
