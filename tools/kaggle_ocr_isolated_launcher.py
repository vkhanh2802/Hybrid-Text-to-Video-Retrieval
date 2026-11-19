"""Run an owned pip-less venv without Kaggle's global sitecustomize hook.

Invoked as: <venv-python> -I -S launcher.py {pip <host-pip-source>|worker <script>} ...
Only that venv's site-packages are enabled. pip source is imported as ONE package,
not by adding the host site-packages directory. No dependencies are suppressed.
"""

import importlib.util
import os
from pathlib import Path
import runpy
import site
import sys


def main():
    if not sys.flags.no_site or not sys.flags.isolated:
        raise RuntimeError('Launcher requires both -I and -S')
    prefix = Path(os.path.abspath(sys.executable)).parent.parent
    if not (prefix / 'pyvenv.cfg').is_file():
        raise RuntimeError('Launcher must use an explicit owned venv interpreter')
    # Python <=3.13 sets venv prefix in site.main(); -S skips it. Restore ONLY
    # the explicit venv prefix, then add its packages without calling site.main.
    sys.prefix = sys.exec_prefix = str(prefix)
    packages = (prefix / 'Lib/site-packages' if os.name == 'nt' else
                prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    if not packages.is_dir():
        raise RuntimeError('Target venv site-packages missing')
    sys.path[:] = [p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]
    site.addsitedir(str(packages))  # Local .pth files only; no global sitecustomize.
    if len(sys.argv) < 3 or sys.argv[1] not in {'pip', 'worker'}:
        raise ValueError('Expected pip source directory or worker script')
    mode, source, arguments = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    if mode == 'pip':
        spec = importlib.util.spec_from_file_location('pip', source / '__init__.py',
                                                     submodule_search_locations=[str(source)])
        if spec is None or spec.loader is None:
            raise RuntimeError('Host pip package source unavailable')
        module = importlib.util.module_from_spec(spec)
        sys.modules['pip'] = module
        spec.loader.exec_module(module)
        sys.argv = ['pip', *arguments]
        runpy.run_module('pip', run_name='__main__')
    else:
        sys.argv = [str(source), *arguments]
        runpy.run_path(str(source), run_name='__main__')


if __name__ == '__main__':
    main()
