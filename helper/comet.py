from comet_ml import Experiment


def init_comet(run_params):
    """
    This function uses the comet_ml package to track the experiment.
    :return: the comet experiment
    """
    # params for Moein
    comet_params = {
        'api_key': "QLZmIFugp5kqZjA4XE2yNS0iZ",
        'project_name': "data_science_project",
        'workspace': "moeinsorkhei"
    }
    tracker = CometTracker(comet_params, run_params)
    return tracker


class CometTracker:
    def __init__(self, comet_params, run_params):
        self.experiment = Experiment(**comet_params)
        self.experiment.log_parameters(run_params)

    def track_metric(self, metric, value):
        self.experiment.log_metric(metric, value)

