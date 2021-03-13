# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines constants used in automated ML in Azure Machine Learning.

Before you begin an experiment, you specify the kind of machine learning problem you are solving
with the :class:`azureml.train.automl.automlconfig.AutoMLConfig` class.
Azure Machine Learning supports task types of classification, regression, and forecasting.
For more information, see [How to define a machine learning
task](https://docs.microsoft.com/azure/machine-learning/service/how-to-define-task-type).

For the task types classification, regression, and forecasing, the supported algorithms are listed,
respectively, in the :class:`azureml.train.automl.constants.SupportedModels.Classification`,
:class:`azureml.train.automl.constants.SupportedModels.Regression`, and
:class:`azureml.train.automl.constants.SupportedModels.Forecasting` classes. The listed algorithms
for each task type are used during the automation and tuning process. As a user, there is no need
for you to specify the algorithm. For more information, see [Configure automated ML experiments in
Python](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-auto-train).
"""

from azureml.automl.core.shared.constants import (
    ModelClassNames,
    MODEL_PATH,
    MODEL_PATH_TRAIN,
    MODEL_PATH_ONNX,
    MODEL_RESOURCE_PATH_ONNX,
    PROPERTY_KEY_OF_MODEL_PATH,
    CHILD_RUNS_SUMMARY_PATH,
    VERIFIER_RESULTS_PATH,
    LOCAL_MODEL_PATH,
    LOCAL_MODEL_PATH_TRAIN,
    LOCAL_MODEL_PATH_ONNX,
    LOCAL_MODEL_RESOURCE_PATH_ONNX,
    LOCAL_CHILD_RUNS_SUMMARY_PATH,
    LOCAL_VERIFIER_RESULTS_PATH,
    EnsembleConstants,
    Defaults,
    RunState,
    API,
    AcquisitionFunction,
    Status,
    PipelineParameterConstraintCheckStatus,
    OptimizerObjectives,
    Optimizer,
    Tasks as CommonTasks,
    ClientErrors,
    ServerStatus,
    TimeConstraintEnforcement,
    PipelineCost,
    Metric,
    MetricObjective,
    TrainingType,
    NumericalDtype,
    TextOrCategoricalDtype,
    TrainingResultsType,
    get_metric_from_type,
    get_status_from_type,
)

from azureml.automl.core.constants import (
    FeatureType, SupportedTransformers, FeaturizationConfigMode
)


AUTOML_SETTINGS_PATH = "automl_settings.pkl"
AUTOML_FIT_PARAMS_PATH = "fit_params.pkl"
LOCAL_SCRIPT_NAME = "_local_managed_startup_script.py"
LOCAL_PREDICT_NAME = "_inference.py"
PREDICT_INPUT_FILE = "predict.pkl"
PREDICTED_METRIC_NAME = "predicted"
MODEL_FILE = "model.pkl"
PYPI_INDEX = 'https://pypi.python.org/simple'
PREDICT_OUTPUT_FILE = "predict_out.pkl"
INFERENCE_OUTPUT = "inference.csv"
MANAGED_RUN_ID_PARAM = "_local_managed_run_id"
SCRIPT_RUN_ID_PROPERTY = "_wrapper_run_id"


class SupportedModels:
    """Defines friendly names for automated ML algorithms supported by Azure Machine Learning.

    If you plan to export your auto ML created models to an
    `ONNX model <https://docs.microsoft.com/azure/machine-learning/concept-onnx>`, only
    those algorithms indicated with an * are able to be converted to the ONNX format.
    Learn more about converting models to
    `ONNX <https://docs.microsoft.com/azure/machine-learning/concept-automated-ml#automl--onnx>`.

    | Classification                      |
    | ------------------------------------|
    | Logistic Regression*                |
    | Light GBM*                          |
    | Gradient Boosting*                  |
    | Decision Tree*                      |
    | K Nearest Neighbors*                |
    | Linear SVC                          |
    | Support Vector Classification (SVC)*|
    | Random Forest*                      |
    | Extremely Randomized Trees*         |
    | Xgboost*                            |
    | Averaged Perceptron Classifier      |
    | Naive* Bayes                        |
    | Stochastic Gradient Descent (SGD)*  |
    | Linear SVM Classifier*              |

    | Regression                          |
    | ----------------------------------- |
    | Elastic Net*                        |
    | Light GBM*                          |
    | Gradient Boosting*                  |
    | Decision Tree*                      |
    | K Nearest Neighbors*                |
    | LARS Lasso*                         |
    | Stochastic Gradient Descent (SGD)   |
    | Random Forest*                      |
    | Extremely Randomized Trees*         |
    | Xgboost*                            |
    | Online Gradient Descent Regressor   |
    | Fast Linear Regressor               |

    | Time Series Forecasting             |
    | ----------------------------------- |
    | Elastic Net                         |
    | Light GBM                           |
    | Gradient Boosting                   |
    | Decision Tree                       |
    | K Nearest Neighbors                 |
    | LARS Lasso                          |
    | Stochastic Gradient Descent (SGD)   |
    | Random Forest                       |
    | Extremely Randomized Trees          |
    | Xgboost                             |
    | Auto-ARIMA                          |
    | Prophet                             |
    | ForecastTCN                         |
    """

    class Classification:
        """Defines the names of classification algorithms used in automated ML.

        Azure supports these classification algorithms, but you as a user do not
        need to specify the algorithms directly. Use the ``allowed_models`` and
        ``blocked_models`` parameters of :class:`azureml.train.automl.automlconfig.AutoMLConfig` class
        to include or exclude models.

        To learn more about in automated ML in Azure see:

        * `What is automated ML <https://docs.microsoft.com/azure/machine-learning/concept-automated-ml>`_

        * `How to define a machine learning
          task <https://docs.microsoft.com/azure/machine-learning/how-to-define-task-type>`_

        * `Configure automated ML experiments in
          Python <https://docs.microsoft.com/azure/machine-learning/how-to-configure-auto-train>`_

        * `TensorFlowDNN, TensorFlowLinearClassifier are deprecated.`
        """

        LogisticRegression = 'LogisticRegression'
        SGDClassifier = 'SGD'
        MultinomialNB = 'MultinomialNaiveBayes'
        BernoulliNB = 'BernoulliNaiveBayes'
        SupportVectorMachine = 'SVM'
        LinearSupportVectorMachine = 'LinearSVM'
        KNearestNeighborsClassifier = 'KNN'
        DecisionTree = 'DecisionTree'
        RandomForest = 'RandomForest'
        ExtraTrees = 'ExtremeRandomTrees'
        LightGBMClassifier = 'LightGBM'
        GradientBoosting = 'GradientBoosting'
        TensorFlowDNNClassifier = 'TensorFlowDNN'
        TensorFlowLinearClassifier = 'TensorFlowLinearClassifier'
        XGBoostClassifier = 'XGBoostClassifier'
        AveragedPerceptronClassifier = 'AveragedPerceptronClassifier'

    class Regression:
        """Defines the names of regression algorithms used in automated ML.

        Azure supports these regression algorithms, but you as a user do not
        need to specify the algorithms directly. Use the ``allowed_models`` and
        ``blocked_models`` parameters of :class:`azureml.train.automl.automlconfig.AutoMLConfig` class
        to include or exclude models.

        To learn more about in automated ML in Azure see:

        * `What is automated ML <https://docs.microsoft.com/azure/machine-learning/concept-automated-ml>`_

        * `How to define a machine learning
          task <https://docs.microsoft.com/azure/machine-learning/how-to-define-task-type>`_

        * `Configure automated ML experiments in
          Python <https://docs.microsoft.com/azure/machine-learning/how-to-configure-auto-train>`_

        * `TensorFlowDNN, TensorFlowLinearRegressor are deprecated.`
        """

        ElasticNet = 'ElasticNet'
        GradientBoostingRegressor = 'GradientBoosting'
        DecisionTreeRegressor = 'DecisionTree'
        KNearestNeighborsRegressor = 'KNN'
        LassoLars = 'LassoLars'
        SGDRegressor = 'SGD'
        RandomForestRegressor = 'RandomForest'
        ExtraTreesRegressor = 'ExtremeRandomTrees'
        LightGBMRegressor = 'LightGBM'
        TensorFlowLinearRegressor = 'TensorFlowLinearRegressor'
        TensorFlowDNNRegressor = 'TensorFlowDNN'
        XGBoostRegressor = 'XGBoostRegressor'
        FastLinearRegressor = 'FastLinearRegressor'
        OnlineGradientDescentRegressor = 'OnlineGradientDescentRegressor'

    class Forecasting(Regression):
        """Defines then names of forecasting algorithms used in automated ML.

        Azure supports these regression algorithms, but you as a user do not
        need to specify the algorithms. Use the ``allowed_models`` and
        ``blocked_models`` parameters of :class:`azureml.train.automl.automlconfig.AutoMLConfig` class
        to include or exclude models.

        To learn more about in automated ML in Azure see:

        * `What is automated ML <https://docs.microsoft.com/azure/machine-learning/concept-automated-ml>`__

        * `How to define a machine learning
          task <https://docs.microsoft.com/azure/machine-learning/how-to-define-task-type>`__

        * `Configure automated ML experiments in
          Python <https://docs.microsoft.com/azure/machine-learning/how-to-configure-auto-train>`__
        """

        AutoArima = 'AutoArima'
        Average = 'Average'
        ExponentialSmoothing = 'ExponentialSmoothing'
        Naive = 'Naive'
        Prophet = 'Prophet'
        SeasonalAverage = 'SeasonalAverage'
        SeasonalNaive = 'SeasonalNaive'
        TCNForecaster = 'TCNForecaster'


MODEL_EXPLANATION_TAG = "model_explanation"

BEST_RUN_ID_SUFFIX = 'best'

MAX_ITERATIONS = 1000
MAX_SAMPLES_AUTOBLOCK = 5000
MAX_SAMPLES_AUTOBLOCKED_ALGOS = [SupportedModels.Classification.KNearestNeighborsClassifier,
                                 SupportedModels.Regression.KNearestNeighborsRegressor,
                                 SupportedModels.Classification.SupportVectorMachine]
EARLY_STOPPING_NUM_LANDMARKS = 20
PIPELINE_FETCH_BATCH_SIZE_LIMIT = 20

DATA_SCRIPT_FILE_NAME = "get_data.py"

"""Names of algorithms that do not support sample weights."""
Sample_Weights_Unsupported = {
    ModelClassNames.RegressionModelClassNames.ElasticNet,
    ModelClassNames.ClassificationModelClassNames.KNearestNeighborsClassifier,
    ModelClassNames.RegressionModelClassNames.KNearestNeighborsRegressor,
    ModelClassNames.RegressionModelClassNames.LassoLars
}

"""Algorithm names that we must force to run in single threaded mode."""
SINGLE_THREADED_ALGORITHMS = [
    ModelClassNames.ClassificationModelClassNames.KNearestNeighborsClassifier,
    ModelClassNames.RegressionModelClassNames.KNearestNeighborsRegressor
]

TrainingType.FULL_SET.remove(TrainingType.TrainValidateTest)


class ComputeTargets:
    """Defines names of compute targets supported in automated ML in Azure Machine Learning.

    Specify the compute target of an experiment run using the :class:`azureml.train.automl.automlconfig.AutoMLConfig`
    class.
    """

    DSVM = 'VirtualMachine'
    BATCHAI = 'BatchAI'
    AMLCOMPUTE = 'AmlCompute'
    LOCAL = 'local'
    ADB = 'ADB'


class TimeSeries:
    """Defines parameters used for time-series forecasting.

    The parameters are specified with the :class:`azureml.train.automl.automlconfig.AutoMLConfig` class.
    The time series forecasting task requires these additional parameters during configuration.

    Attributes:
        DROP_COLUMN_NAMES: Defines the names of columns to drop from featurization.

        GRAIN_COLUMN_NAMES: Defines the names of columns that contain individual time series data in your training
            data.

        MAX_HORIZON: Defines the length of time to predict out based on the periodicity of the data.

        TIME_COLUMN_NAME: Defines the name of the column in your training data containing a valid time-series.

    """

    TIME_COLUMN_NAME = 'time_column_name'
    GRAIN_COLUMN_NAMES = 'grain_column_names'
    DROP_COLUMN_NAMES = 'drop_column_names'
    MAX_HORIZON = 'max_horizon'


class Tasks(CommonTasks):
    """A subclass of Tasks in common.core module that can be extended to add more task types for the SDK.

    You can set the task type for your automated ML experiments using the ``task`` parameter of the
    :class:`azureml.train.automl.automlconfig.AutoMLConfig` constructor. For more information about
    tasks, see `How to define a machine learning
    task <https://docs.microsoft.com/azure/machine-learning/how-to-define-task-type>`_.
    """

    CLASSIFICATION = CommonTasks.CLASSIFICATION
    REGRESSION = CommonTasks.REGRESSION
    FORECASTING = 'forecasting'
    IMAGE_CLASSIFICATION = CommonTasks.IMAGE_CLASSIFICATION
    IMAGE_MULTI_LABEL_CLASSIFICATION = CommonTasks.IMAGE_MULTI_LABEL_CLASSIFICATION
    IMAGE_OBJECT_DETECTION = CommonTasks.IMAGE_OBJECT_DETECTION
    IMAGE_INSTANCE_SEGMENTATION = CommonTasks.IMAGE_INSTANCE_SEGMENTATION
    ALL_IMAGE = CommonTasks.ALL_IMAGE
    TEXT_CLASSIFICATION_MULTILABEL = CommonTasks.TEXT_CLASSIFICATION_MULTILABEL
    ALL_TEXT = CommonTasks.ALL_TEXT
    ALL_DNN = CommonTasks.ALL_DNN
    ALL = [CommonTasks.CLASSIFICATION, CommonTasks.REGRESSION, FORECASTING] + ALL_IMAGE + ALL_TEXT


class ExperimentObserver:
    """Constants used by the Experiment Observer to report progress during preprocessing."""

    EXPERIMENT_STATUS_METRIC_NAME = "experiment_status"
    EXPERIMENT_STATUS_DESCRIPTION_METRIC_NAME = "experiment_status_description"


class Framework:
    """Constants for the various supported framework."""

    PYTHON = "python"
    PYSPARK = "pyspark"
    FULL_SET = {"python", "pyspark"}


class Scenarios:
    """Constants for the various curated environment scenarios."""

    SDK = "SDK"  # Default for azureml.train.automl.VESRION<1.5.0
    SDK_COMPATIBLE = "SDK-1.13.0"  # Default for PROD SDK
    SDK_COMPATIBLE_1120 = "SDK-Compatible"  # Default for azureml.train.automl.VESRION>1.5.0,<1.13.0
    _NON_PROD = "non-prod"


class Environments:
    """Curated environments defined for AutoML."""

    AUTOML = "AzureML-AutoML"
    AUTOML_DNN = "AzureML-AutoML-DNN"
    AUTOML_GPU = "AzureML-AutoML-GPU"
    AUTOML_DNN_GPU = "AzureML-AutoML-DNN-GPU"


class _DataArgNames:
    X = "X"
    y = "y"
    sample_weight = "sample_weight"
    X_valid = "X_valid"
    y_valid = "y_valid"
    sample_weight_valid = "sample_weight_valid"
    training_data = "training_data"
    validation_data = "validation_data"
    test_data = "test_data"


class SupportedInputDatatypes:
    """Input data types supported by AutoML for different Run types."""

    PANDAS = "pandas.DataFrame"
    TABULAR_DATASET = "azureml.data.tabular_dataset.TabularDataset"
    PIPELINE_OUTPUT_TABULAR_DATASET = "azureml.pipeline.core.pipeline_output_dataset.PipelineOutputTabularDataset"

    LOCAL_RUN_SCENARIO = [PANDAS, TABULAR_DATASET, PIPELINE_OUTPUT_TABULAR_DATASET]
    REMOTE_RUN_SCENARIO = [TABULAR_DATASET, PIPELINE_OUTPUT_TABULAR_DATASET]
    ALL = [PANDAS, TABULAR_DATASET, PIPELINE_OUTPUT_TABULAR_DATASET]
