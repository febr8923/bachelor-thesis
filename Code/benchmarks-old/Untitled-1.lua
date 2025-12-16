
1) in uenv: (rmm_dev) (ml-venv) fbrunne@clariden-ln004:~/rmm> ./build.sh librmm rmm -> RESOLVED (installed dependencies)
ERROR: Exception:
Traceback (most recent call last):
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 105, in _run_wrapper
    status = _inner_run()
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 96, in _inner_run
    return self.run(options, args)
           ~~~~~~~~^^^^^^^^^^^^^^^
  File "/users/fbrunne/projectspip install --upgrade rapids-build-backend setuptools/ml-venv/lib/python3.13/site-packages/pip/_internal/cli/req_command.py", line 68, in wrapper
    return func(self, options, args)
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/commands/install.py", line 387, in run
    requirement_set = resolver.resolve(
        reqs, check_supported_wheels=not options.target_dir
    )
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 77, in resolve
    collected = self.factory.collect_root_requirements(root_reqs)
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 545, in collect_root_requirements
    reqs = list(
        self._make_requirements_from_install_req(
    ...<2 lines>...
        )
    )
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 501, in _make_requirements_from_install_req
    cand = self._make_base_candidate_from_link(
        ireq.link,
    ...<2 lines>...
        version=None,
    )
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 233, in _make_base_candidate_from_link
    self._link_candidate_cache[link] = LinkCandidate(
                                       ~~~~~~~~~~~~~^
        link,
        ^^^^^
    ...<3 lines>...
        version=version,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 306, in __init__
    super().__init__(
    ~~~~~~~~~~~~~~~~^
        link=link,
        ^^^^^^^^^^
    ...<4 lines>...
        version=version,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 159, in __init__
    self.dist = self._prepare()
                ~~~~~~~~~~~~~^^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 236, in _prepare
    dist = self._prepare_distribution()
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 317, in _prepare_distribution
    return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 532, in prepare_linked_requirement
    return self._prepare_linked_requirement(req, parallel_builds)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 647, in _prepare_linked_requirement
    dist = _get_prepared_distribution(
        req,
    ...<3 lines>...
        self.check_build_deps,
    )
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 71, in _get_prepared_distribution
    abstract_dist.prepare_distribution_metadata(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        finder, build_isolation, check_build_deps
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 69, in prepare_distribution_metadata
    self.req.prepare_metadata()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/req/req_install.py", line 575, in prepare_metadata
    self.metadata_directory = generate_metadata(
                              ~~~~~~~~~~~~~~~~~^
        build_env=self.build_env,
        ^^^^^^^^^^^^^^^^^^^^^^^^^
        backend=self.pep517_backend,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        details=details,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/operations/build/metadata.py", line 34, in generate_metadata
    distinfo_dir = backend.prepare_metadata_for_build_wheel(metadata_dir)
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_internal/utils/misc.py", line 723, in prepare_metadata_for_build_wheel
    return super().prepare_metadata_for_build_wheel(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        metadata_directory=metadata_directory,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        config_settings=cs,
        ^^^^^^^^^^^^^^^^^^^
        _allow_fallback=_allow_fallback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 224, in prepare_metadata_for_build_wheel
    return self._call_hook(
           ~~~~~~~~~~~~~~~^
        "prepare_metadata_for_build_wheel",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        },
        ^^
    )
    ^
  File "/users/fbrunne/projects/ml-venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 402, in _call_hook
    raise BackendUnavailable(
    ...<4 lines>...
    )
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'rapids_build_backend.build'

2)
Installing collected packages: numpy, librmm-cu13, rmm-cu13
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
pytorch-transformers 1.2.0 requires regex, which is not installed.
pytorch-transformers 1.2.0 requires requests, which is not installed.
pytorch-transformers 1.2.0 requires sentencepiece, which is not installed.
pytorch-transformers 1.2.0 requires torch>=1.0.0, which is not installed.
pytorch-transformers 1.2.0 requires tqdm, which is not installed.
rmm 25.10.0 requires librmm==25.10.*,>=0.0.0a0, which is not installed.
rmm-cu12 25.10.0 requires cuda-python<13.0a0,>=12.9.2, but you have cuda-python 13.0.1 which is incompatible.
Successfully installed librmm-cu13-25.10.0 numpy-2.3.2 rmm-cu13-25.10.0

3)
(rmm_dev) (ml-venv) fbrunne@clariden-ln004:~/rmm> pytest -v
========================================================== test session starts ==========================================================
platform linux -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0 -- /users/fbrunne/miniconda3/envs/rmm_dev/bin/python3.13
cachedir: .pytest_cache
rootdir: /users/fbrunne/rmm
configfile: pyproject.toml
plugins: cov-6.2.1
collected 0 items / 1 error                                                                                                             

================================================================ ERRORS =================================================================
_________________________________________________ ERROR collecting python/rmm/rmm/tests _________________________________________________
../miniconda3/envs/rmm_dev/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
../miniconda3/envs/rmm_dev/lib/python3.13/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
python/rmm/rmm/tests/conftest.py:17: in <module>
    import rmm
E   ModuleNotFoundError: No module named 'rmm'
======================================================== short test summary info ========================================================
ERROR python/rmm/rmm/tests - ModuleNotFoundError: No module named 'rmm'
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
=========================================================== 1 error in 0.34s ============================================================

4) 
(rmm_dev) fbrunne@clariden-ln004:~/rmm> pytest -v
========================================================== test session starts ==========================================================
platform linux -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0 -- /users/fbrunne/miniconda3/envs/rmm_dev/bin/python3.13
cachedir: .pytest_cache
rootdir: /users/fbrunne/rmm
configfile: pyproject.toml
plugins: cov-6.2.1
collected 0 items / 1 error                                                                                                             

================================================================ ERRORS =================================================================
_________________________________________________ ERROR collecting python/rmm/rmm/tests _________________________________________________
../miniconda3/envs/rmm_dev/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
../miniconda3/envs/rmm_dev/lib/python3.13/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
python/rmm/rmm/tests/conftest.py:17: in <module>
    import rmm
python/rmm/rmm/__init__.py:26: in <module>
    from rmm import mr
python/rmm/rmm/mr.py:14: in <module>
    from rmm.pylibrmm.memory_resource import (
python/rmm/rmm/pylibrmm/__init__.py:15: in <module>
    from .device_buffer import DeviceBuffer
rmm/pylibrmm/device_buffer.pyx:1: in init rmm.pylibrmm.device_buffer
    ???
E   ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /users/fbrunne/miniconda3/envs/rmm_dev/lib/python3.13/site-packages/rmm/pylibrmm/memory_resource.cpython-313-aarch64-linux-gnu.so)
======================================================== short test summary info ========================================================
ERROR python/rmm/rmm/tests - ImportError: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /users/fbrunne/miniconda3/envs/rmm_dev/lib/py...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
=========================================================== 1 error in 0.28s =========================================================

5)
========================================================== test session starts ==========================================================
platform linux -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0 -- /users/fbrunne/miniconda3/envs/rmm_dev/bin/python3.13
cachedir: .pytest_cache
rootdir: /users/fbrunne/rmm
configfile: pyproject.toml
plugins: cov-6.2.1
collected 7 items / 1 error / 1 skipped                                                                                                 

================================================================ ERRORS =================================================================
___________________________________________ ERROR collecting python/rmm/rmm/tests/test_rmm.py ___________________________________________
python/rmm/rmm/tests/test_rmm.py:38: in <module>
    rmm._cuda.gpu.getDevice(),
    ^^^^^^^^^^^^^^^^^^^^^^^^^
python/rmm/rmm/_cuda/gpu.py:58: in getDevice
    raise CUDARuntimeError(status)
E   rmm._cuda.gpu.CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
======================================================== short test summary info ========================================================
ERROR python/rmm/rmm/tests/test_rmm.py - rmm._cuda.gpu.CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
====================================================== 1 skipped, 1 error in 2.95s ======================================================