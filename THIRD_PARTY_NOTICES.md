# Third-Party Notices

This repository depends on third-party open-source packages and also includes a
small amount of third-party demo material for testing and reference.

This file is an informational notice for repository users. It does not replace
the original license texts or upstream terms.

## Runtime dependencies

The project declares its main Python dependencies in
[`Code/pyproject.toml`](Code/pyproject.toml) and
[`requirements.txt`](requirements.txt).

Core dependencies used directly by the project include:

- `shap`
  - Upstream: <https://github.com/shap/shap>
  - License: MIT
- `xgboost`
  - Upstream: <https://github.com/dmlc/xgboost>
  - License: Apache-2.0
- `geoxgboost`
  - Upstream: <https://pypi.org/project/geoxgboost/>
  - License: MIT
- `scikit-learn`
  - Upstream: <https://github.com/scikit-learn/scikit-learn>
  - License: BSD-3-Clause
- `pandas`
  - Upstream: <https://github.com/pandas-dev/pandas>
  - License: BSD-3-Clause
- `numpy`
  - Upstream: <https://github.com/numpy/numpy>
  - License: BSD-3-Clause
- `matplotlib`
  - Upstream: <https://github.com/matplotlib/matplotlib>
  - License: Matplotlib License (BSD-style)
- `scipy`
  - Upstream: <https://github.com/scipy/scipy>
  - License: BSD-3-Clause

Unless otherwise stated, these dependencies are used as external packages and
are not vendored into this repository as source code.

## Included third-party demo material

The repository includes the following third-party demo folder:

- [`test/DemoGXGBoost/`](test/DemoGXGBoost/)
  - Origin: <https://github.com/geogreko/DemoGXGBoost/tree/main>
  - Included upstream files retain their own license and attribution
  - Local license file:
    [`test/DemoGXGBoost/LICENSE`](test/DemoGXGBoost/LICENSE)
  - Upstream license: MIT

This folder also includes demo data and tutorial material distributed as part
of the upstream demo package. Users should review the upstream repository and
the retained local license file before reusing or redistributing that material.

## Repository license vs. third-party licenses

The root-level [`LICENSE`](LICENSE) applies to
the original code and documentation in this repository unless a more specific
license notice applies to a subdirectory or file.

Third-party code, data, documentation, and other assets remain under their own
respective licenses.

## Practical guidance

- If you reuse only the original code from this repository, follow the root
  MIT license.
- If you reuse files from third-party subfolders, preserve the upstream
  notices that already ship with those materials.
- If you redistribute bundled third-party packages or vendored code in the
  future, you may need to include their full license texts and any required
  notices in your distribution.
