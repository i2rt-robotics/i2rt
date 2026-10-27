Upstream SOEM v1.4.0 (https://github.com/OpenEtherCATsociety/SOEM/releases/tag/v1.4.0), release
tarball sha256 `e58e1e0c216dade84a76d2c10ab34d2fa156ca639eac00a8081be6e88911d86e`. Only `LICENSE`,
the core (`soem/`) and the Linux `osal`/`oshw` layers are kept, byte-for-byte unmodified; they are
compiled by `../../CMakeLists.txt`, not by SOEM's own build.

Licence: GPLv2 with a linking exception (see `LICENSE`) -- linking it does not bring the rest of
this package under the GPL, but this source must stay available, which vendoring it does. v2.0 and
later are GPLv3/commercial and drop the `ec_` API used here: do not upgrade without a licence review.
