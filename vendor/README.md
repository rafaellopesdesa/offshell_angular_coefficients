# Vendored `nsbi-common-utils` wheel

`nsbi_common_utils-0.1.dev794-py3-none-any.whl` is a pure-Python wheel built
from the IRIS-HEP `nsbi-lhc-toolkit` repository at the exact commit

```text
fc09848fc6540fd32310faebbe9db6eea7ecd17b
```

Upstream source: <https://github.com/iris-hep/nsbi-lhc-toolkit>

The upstream repository is licensed under the MIT License.  The upstream
license file is included inside the wheel at
`nsbi_common_utils-0.1.dev794.dist-info/licenses/LICENSE.txt`.

## Integrity

```text
SHA256  bb3df9165503ae5e99437873609aa7e7df424589c6490b25f63020123b3e5e85
```

The wheel contains only `src/nsbi_common_utils`, its JSON schema, package
metadata, and the upstream license.  It does not contain the LFS-managed
example datasets, plots, or trained models from the upstream repository.

## Rebuild

The empty Git filters below deliberately leave unrelated LFS files as pointer
text during checkout.  They are not part of the Python wheel.

```bash
git -c filter.lfs.process= \
    -c filter.lfs.smudge= \
    -c filter.lfs.required=false \
    clone https://github.com/iris-hep/nsbi-lhc-toolkit.git nsbi-lhc-toolkit

cd nsbi-lhc-toolkit
git -c filter.lfs.process= \
    -c filter.lfs.smudge= \
    -c filter.lfs.required=false \
    checkout --detach fc09848fc6540fd32310faebbe9db6eea7ecd17b

python -m pip wheel . --wheel-dir dist --no-deps
sha256sum dist/nsbi_common_utils-0.1.dev794-py3-none-any.whl
```
