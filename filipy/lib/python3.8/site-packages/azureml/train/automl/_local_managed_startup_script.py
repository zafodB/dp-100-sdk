# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import argparse
import logging
import os
import pickle as pkl

from azureml._common.exceptions import AzureMLException
from azureml._history.utils.constants import LOGS_AZUREML_DIR
from azureml._restclient.constants import RunStatus
from azureml.automl.core import dataset_utilities
from azureml.automl.core._run import run_lifecycle_utilities
from azureml.automl.core._logging import log_server
from azureml.automl.core.shared import logging_utilities
from azureml.core import Run
from azureml.core.runconfig import RunConfiguration
from azureml.train.automl import _logging
from azureml.train.automl.constants import AUTOML_SETTINGS_PATH
from azureml.train.automl._azureautomlsettings import AzureAutoMLSettings
from azureml.train.automl._azure_experiment_state import AzureExperimentState
from azureml.train.automl._local_managed_utils import get_data
from azureml.train.automl.constants import _DataArgNames
try:
    from azureml.train.automl.runtime._remote_script import local_managed_wrapper
    has_wrapper = True
except ImportError:
    has_wrapper = False

logger = logging.getLogger('azureml.train.automl._script')


if __name__ == '__main__':
    # Note: this file is not intended to be run manually, it is only used for submitting local managed runs
    with log_server.new_log_context():
        if has_wrapper:
            local_managed_wrapper()
        else:
            # This is legacy code that shouldn't be in use anywhere anymore. Will deprecate after 1.23 CE is out
            # VSO: 1046043
            run = Run.get_context()

            logger.info("Starting local managed execution.")
            try:

                parser = argparse.ArgumentParser()

                parser.add_argument('--{}'.format(_DataArgNames.X), type=str,
                                    dest=_DataArgNames.X)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.X), type=str,
                                    dest="{}_dtype".format(_DataArgNames.X))
                parser.add_argument('--{}'.format(_DataArgNames.y), type=str,
                                    dest=_DataArgNames.y)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.y), type=str,
                                    dest="{}_dtype".format(_DataArgNames.y))
                parser.add_argument('--{}'.format(_DataArgNames.sample_weight), type=str,
                                    dest=_DataArgNames.sample_weight)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.sample_weight), type=str,
                                    dest="{}_dtype".format(_DataArgNames.sample_weight))
                parser.add_argument('--{}'.format(_DataArgNames.X_valid), type=str,
                                    dest=_DataArgNames.X_valid)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.X_valid), type=str,
                                    dest="{}_dtype".format(_DataArgNames.X_valid))
                parser.add_argument('--{}'.format(_DataArgNames.y_valid), type=str,
                                    dest=_DataArgNames.y_valid)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.y_valid), type=str,
                                    dest="{}_dtype".format(_DataArgNames.y_valid))
                parser.add_argument('--{}'.format(_DataArgNames.sample_weight_valid), type=str,
                                    dest=_DataArgNames.sample_weight_valid)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.sample_weight_valid), type=str,
                                    dest="{}_dtype".format(_DataArgNames.sample_weight_valid))
                parser.add_argument('--{}'.format(_DataArgNames.training_data), type=str,
                                    dest=_DataArgNames.training_data)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.training_data), type=str,
                                    dest="{}_dtype".format(_DataArgNames.training_data))
                parser.add_argument('--{}'.format(_DataArgNames.validation_data), type=str,
                                    dest=_DataArgNames.validation_data)
                parser.add_argument('--{}-dtype'.format(_DataArgNames.validation_data), type=str,
                                    dest="{}_dtype".format(_DataArgNames.validation_data))
                args = parser.parse_args()

                logger.info("Unpickling settings for local managed.")
                with open(AUTOML_SETTINGS_PATH, 'rb+') as f:
                    automl_setting = pkl.load(f)

                experiment = run.experiment
                ws = experiment.workspace

                # This is blocking for local anyways, and this will print to driver logs
                automl_setting["show_output"] = True
                automl_setting['path'] = '.'
                automl_setting['debug_log'] = os.path.join(LOGS_AZUREML_DIR, 'automl.log')
                settings = AzureAutoMLSettings(experiment, **automl_setting)

                _logging.set_run_custom_dimensions(automl_settings=settings,
                                                   parent_run_id=settings._local_managed_run_id,
                                                   child_run_id=run.id)

                logger.info("Fetching data for local managed.")

                # These steps no longer happen in the driver, so we need to manually change the data
                X, y, sample_weight, X_valid, y_valid, sample_weight_valid = dataset_utilities.convert_inputs(
                    get_data(workspace=ws, location=args.X, dtype=args.X_dtype),
                    get_data(workspace=ws, location=args.y, dtype=args.y_dtype),
                    get_data(workspace=ws, location=args.sample_weight, dtype=args.sample_weight_dtype),
                    get_data(workspace=ws, location=args.X_valid, dtype=args.X_valid_dtype),
                    get_data(workspace=ws, location=args.y_valid, dtype=args.y_valid_dtype),
                    get_data(workspace=ws, location=args.sample_weight_valid, dtype=args.sample_weight_valid_dtype)
                )

                training_data, validation_data, test_data = dataset_utilities.convert_inputs_dataset(
                    get_data(workspace=ws, location=args.training_data, dtype=args.training_data_dtype),
                    get_data(workspace=ws, location=args.validation_data, dtype=args.validation_data_dtype),
                    None  # get_data(workspace=ws, location=args.test_data, dtype=args.test_data_dtype)
                )

                fit_params = {
                    "run_configuration": RunConfiguration(),
                    "_script_run": run,
                    _DataArgNames.X: X,
                    _DataArgNames.y: y,
                    _DataArgNames.sample_weight: sample_weight,
                    _DataArgNames.X_valid: X_valid,
                    _DataArgNames.y_valid: y_valid,
                    _DataArgNames.sample_weight_valid: sample_weight_valid,
                    _DataArgNames.training_data: training_data,
                    _DataArgNames.validation_data: validation_data
                }

                from azureml.train.automl.runtime._experiment_drivers.local_experiment_driver import \
                    LocalExperimentDriver

                experiment_state = AzureExperimentState(experiment, settings)
                driver = LocalExperimentDriver(experiment_state)

                logger.info("Starting a local legacy fit inside local managed scenario.")
                try:
                    local_run = driver.start(**fit_params)
                except AzureMLException as e:
                    # This should all be handled internally, but in case a run is still running and an azureml
                    # exception was returned, then fail the parent but succeed the setup script.
                    local_run = Run(experiment, settings._local_managed_run_id)
                    if local_run not in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED]:
                        run_lifecycle_utilities.fail_run(local_run, e, is_aml_compute=False)
            except Exception as e:
                logging_utilities.log_traceback(e, logger)
                raise
