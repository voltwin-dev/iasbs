# CuAu cluster-expansion energy: exact provenance

The CuAu benchmark uses a **published DFT-fitted cluster expansion** as its energy
oracle.  Nothing in this repository refits it, and no DFT is ever run here.

## Upstream source

```text
repository   https://github.com/xiaochendu/metadns
commit       ae46941ff7a3027c87ca6a6bd97c5b0064e06c8c   (see METADNS_COMMIT.txt)
commit date  2026-05-20 14:30:01 -0400
paper        MetaDNS, ICML 2026 -- https://openreview.net/forum?id=OY7Qe2ZSx9
zenodo       https://doi.org/10.5281/zenodo.20301979   (optional, not used)
```

The clone lives in `external/metadns/` and is **gitignored**; only the three
small input files below are vendored into this repository.

## Vendored files

```text
data/cuau/cuau_fcc_4x4x4_supercell.vasp          64 FCC sites  (CuAu-M, headline)
data/cuau/cuau_fcc_2x2x4_supercell.vasp          16 FCC sites  (CuAu-S, exact)
data/cuau/CI_params_ECI_CuAu_Final_Submission.json   12 ECIs
```

SHA256 sums are in `SHA256SUMS.txt`.  Verify with

```bash
sha256sum -c data/cuau/SHA256SUMS.txt
```

## Reproducing the download

```bash
mkdir -p external data/cuau
git clone https://github.com/xiaochendu/metadns.git external/metadns
git -C external/metadns checkout ae46941ff7a3027c87ca6a6bd97c5b0064e06c8c
git -C external/metadns rev-parse HEAD | tee data/cuau/METADNS_COMMIT.txt
cp external/metadns/data/cuau/cuau_fcc_4x4x4_supercell.vasp \
   external/metadns/data/cuau/cuau_fcc_2x2x4_supercell.vasp \
   external/metadns/data/cuau/CI_params_ECI_CuAu_Final_Submission.json \
   data/cuau/
sha256sum data/cuau/*.vasp data/cuau/*.json | tee data/cuau/SHA256SUMS.txt
```

## Cell geometry (measured, not assumed)

Both VASP files use **Cartesian** coordinates and FCC *primitive* cell vectors
with conventional lattice constant `a = 3.8 A`:

```text
4x4x4   a1 = a*(0,1,1)  a2 = a*(1,0,1)  a3 = a*(1,1,0)      -> 64 sites
2x2x4   a1 = (a/2)*(0,2,2)/1 ... 2x2x4 repeats of primitive -> 16 sites
```

Volume per site is `a^3/4 = 13.718 A^3` in both files, confirming a clean
commensurate FCC lattice with no relaxation.  The `4x4x4` cell is an equal
`4 x 4 x 4` repeat of the primitive cell and therefore admits the cubic
`x <-> y <-> z` point-group operations used by the orientation-symmetry gate;
the `2x2x4` cell does **not** (2,2,4 repeats), so no exact `1/3` orientation
mass may be claimed there.

The three L10 X-point wavevectors `q = (2*pi/a) * e_alpha` are commensurate with
both cells (`q . a_i` is a multiple of `2*pi` for every cell vector), so the
structure-factor order parameters are well defined without any rotation of the
q-vectors.

## Energy model

```text
crystalstructure  fcc
a                 3.8
concentration     Au, Cu  (basis_elements=[["Au","Cu"]])
max_cluster_dia   [6.0, 4.5, 4.5]
ECIs              c0, c1_0, 4 pairs, 2 triplets, 3 quadruplets  (12 total)
backend           clease.settings.CEBulk + icet ClusterExpansionCalculator
```

**Unit warning.** The upstream `ClusterExpansionModel` in
`external/metadns/energy_cuau.py` documents that its `energy` methods return
values in units of `k_B*T` (`energy_conversion_factor = ase.units.eV /
ase.units.kB`).  The IASBS CuAu adapter must use the **raw total configurational
energy in eV for the whole supercell**, and the energy-parity gate is run against
that raw quantity.  Never infer the unit from magnitude.

At fixed equiatomic composition a semi-grand chemical-potential term is constant
across the sector and cancels from every target ratio, so the IASBS target uses
`E_config` only.

## Environment

The CE stack pins `numpy < 2` in practice, while this repository's original
`requirements.txt` pins NumPy 2.x.  The CuAu work therefore lives in a
**separate** conda environment and `requirements.txt` is left untouched:

```text
env name   cuau_env
python     3.11
torch      2.5.1
numpy      < 2
ase, clease == 1.2.0, icet, mchammer-pt == 0.27.1
```

Exact resolved versions are recorded in `requirements-cuau.txt` and in every
result JSON.
