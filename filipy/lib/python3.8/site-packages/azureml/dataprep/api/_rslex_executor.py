import json
import threading
from .engineapi.api import get_engine_api
from ._loggerfactory import _LoggerFactory, session_id, instrumentation_key, log_directory, verbosity, HBI_MODE


log = _LoggerFactory.get_logger('rslex_executor')


class _RsLexExecutor:
    def __init__(self):
        self._pending_executions = {}

    def await_result(self, execution_id, callback, fail_on_error, fail_on_mixed_types, fail_on_out_of_range_datetime):
        self._pending_executions[execution_id] = \
            (callback, fail_on_error, fail_on_mixed_types, fail_on_out_of_range_datetime)

    def _load_pending_execution(self, execution_id):
        if execution_id is None:
            return None, False, False, False
        else:
            return self._pending_executions.pop(execution_id)

    def execute_rslex_script(self, request, writer):
        try:
            from azureml.dataprep.rslex import Executor
            ex = Executor()
            script = request.get('script')
            execution_id = request.get('executionId')
            traceparent = request.get('traceparent', '')
            callback, fail_on_error, fail_on_mixed_types, fail_on_out_of_range_datetime = \
                self._load_pending_execution(execution_id)

            error = None
            (batches, num_partitions) = ex.execute(
                script,
                callback is not None,
                fail_on_error,
                fail_on_mixed_types,
                fail_on_out_of_range_datetime,
                traceparent
            )
            if callback is not None:
                callback(batches)
            writer.write(json.dumps({'result': 'success', 'partitionCount': num_partitions}) + '\n')
        except Exception as e:
            error = repr(e)
        finally:
            if error is not None:
                writer.write(json.dumps({'result': 'error', 'error': error}) + '\n')
                log.info('Execution failed with rslex.\nFallback to clex.')


_rslex_executor = None
_rslex_environment_init = False
_rslex_environment_lock = threading.Lock()


def get_rslex_executor():
    global _rslex_executor
    if _rslex_executor is None:
        _rslex_executor = _RsLexExecutor()
    ensure_rslex_environment()

    return _rslex_executor


def ensure_rslex_environment(caller_session_id: str = None):
    global _rslex_environment_init
    if _rslex_environment_init is False:
        try:
            # Acquire lock on mutable access to _rslex_environment_init
            _rslex_environment_lock.acquire()
            # Was _rslex_environment_init set while we held the lock?
            if _rslex_environment_init is True:
                return _rslex_environment_init
            # Initialize new RsLex Environment
            import atexit
            import azureml.dataprep.rslex as rslex
            engine_api = get_engine_api()
            run_info = _LoggerFactory._try_get_run_info()
            rslex.init_environment(
                engine_api._engine_server_port,
                engine_api._engine_server_secret,
                log_directory,
                instrumentation_key,
                verbosity,
                HBI_MODE,
                session_id,
                caller_session_id,
                json.dumps(run_info) if run_info is not None else None
            )
            _rslex_environment_init = True
            _LoggerFactory.add_default_custom_dimensions({'rslex_version': rslex.__version__})
            atexit.register(rslex.release_environment)
        except Exception as e:
            log.error('ensure_rslex_environment failed with {}'.format(e))
            raise
        finally:
            if _rslex_environment_lock.locked():
                _rslex_environment_lock.release()
    return _rslex_environment_init


def use_rust_execution(use: bool):
    get_engine_api().use_rust_execution(use)
    if use is True:
        ensure_rslex_environment()


def ensure_rslex_handler(requests_channel):
    requests_channel.register_handler('execute_rslex_script',
                                      lambda r, w, _: get_rslex_executor().execute_rslex_script(r, w))
