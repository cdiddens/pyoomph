# preCICE: the .deb must match the distribution release

Written 2026-08-03 on `duarte`, after the nightly had reported the two `PreCICE_Coupling` tutorials
as failures every single night since it was set up.

## The symptom

```
ModuleNotFoundError: No module named 'precice'
```

and, once `pip install pyprecice` has been run,

```
ImportError: libboost_log_setup.so.1.90.0: cannot open shared object file: No such file or directory
```

The second one is the informative one, and it is **not** a problem with the Python bindings.

## The cause

preCICE publishes one `.deb` per distribution release, all with the same version number:

    libprecice3_3.4.1_jammy.deb  libprecice3_3.4.1_noble.deb
    libprecice3_3.4.1_trixie.deb libprecice3_3.4.1_resolute.deb

`duarte` (Ubuntu 24.04, `noble`) had the **`resolute`** one installed. dpkg accepts it — the package
declares only `libboost-dev (>= 1.65)`, `python3-dev`, `petsc-dev (>= 3.6)` and so on, with no upper
bounds and no ABI versions — but the binary inside was compiled against a much newer userland:

```
$ ldd /usr/lib/x86_64-linux-gnu/libprecice.so.3
  libm.so.6: version `GLIBC_2.43' not found     # noble has 2.39
  libboost_log_setup.so.1.90.0 => not found     # noble has 1.83.0
  libxml2.so.16 => not found
  libpetsc_real.so.3.24 => not found
  libpython3.14.so.1.0 => not found             # noble has 3.12
```

So `libprecice.so.3` could not be loaded at all, and no amount of work on the Python side would have
helped. `apt-cache policy libprecice3` gives it away: the version table lists only
`/var/lib/dpkg/status` at priority 100, i.e. it came from a downloaded file and no configured repo
provides it.

## The fix

```
. /etc/os-release && echo "$VERSION_CODENAME"          # noble
curl -LO https://github.com/precice/precice/releases/download/v3.4.1/libprecice3_3.4.1_noble.deb
sudo apt install ./libprecice3_3.4.1_noble.deb
pip install --user pyprecice                            # 3.4.0; major version must match libprecice
```

`pyprecice` does **not** need rebuilding when libprecice is swapped, as long as the soname is
unchanged: its extension links only `libprecice.so.3`, `libstdc++`, `libgcc_s` and `libc`, and every
boost / glibc / libpython dependency is transitive through libprecice.

Verify with the version string rather than a bare import, because that is what exercises the C++
library:

```
python3 -c "import precice; print(precice.get_version_information())"
```

## What the tutorial pipeline does and does not check

Run with no arguments — which is how `citools/test_all_tutorial_scripts.py` runs everything — both
scripts take the `precice_participant==""` branch and solve the **monolithic** domain. preCICE is
imported and then never used. A green pipeline therefore says nothing about the coupling, which is
what the runner's closing line ("please check e.g. preCICE runs manually") has always meant.

The coupled case has to be started as two participants sharing a working directory:

```
cd <dir containing precice-config.xml>
python3 partitioned_heat_conduction.py -P precice_participant=Dirichlet --outdir cpl_D &
python3 partitioned_heat_conduction.py -P precice_participant=Neumann   --outdir cpl_N
```

`--outdir` is what keeps them apart; without it both write the same output directory. Checked on
2026-08-03 for both tutorials: each participant reaches "final time-window: 10, final time: 1" and
closes its channels, rc 0 on both sides.

If a machine cannot run preCICE, the runner reports the two scripts as skips rather than failures and
names the reason — including the unloadable-library case above, which is an `ImportError` and not a
`ModuleNotFoundError`. See `missing_optional_module()` there.
