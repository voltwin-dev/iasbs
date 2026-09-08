# GB1 four-site combinatorial fitness landscape

Source: Wu N.C., Dai L., Olson C.A., Lloyd-Smith J.O., Sun R. (2016),
*"Adaptation in protein fitness landscapes is facilitated by indirect paths"*,
**eLife 5:e16965**, DOI [10.7554/eLife.16965](https://doi.org/10.7554/eLife.16965).
Licensed CC-BY 4.0.

The two supplementary workbooks are **not** committed (see `.gitignore`); they
are ~9 MB and are redistributable only from the publisher.  Fetch them with:

```bash
mkdir -p data/gb1
curl -L -o data/gb1/elife-16965-supp1-v4.xlsx \
  https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx
curl -L -o data/gb1/elife-16965-supp2-v4.xlsx \
  https://cdn.elifesciences.org/articles/16965/elife-16965-supp2-v4.xlsx
```

## What they contain

| file | content | rows |
|---|---|---|
| `supp1` | directly measured variants (`Variants`, `Fitness`) | 149,361 |
| `supp2` | imputed variants (regression-filled) | 10,639 |
| union | all 20^4 sequences at sites V39, D40, G41, V54 | 160,000 |

The two sets are disjoint; their union is exactly complete.  Wild type is
`VDGV` with fitness 1.0 by construction.

## How the experiment uses them

`structured_asbs/fixed_support.py::load_gb1` merges both sheets into a dense
`(20,20,20,20)` fitness array.  The fixed-support target restricts to the
Hamming-distance-3 shell around the wild type,

```
X^(20)_{4,3} = { x in {0..19}^4 : #{i : x_i != 0} = 3 },   |X| = C(4,3)*19^3 = 27,436,
```

where label `0` means "wild-type residue at this site".  The energy is
`E(x) = -log(F(x) + 1e-4)` and the target law is `pi ∝ exp(-E/tau)` with
`tau = 1.0`.

Reproduce the loader check (test `[A2.14]`) with:

```bash
python structured_asbs/tests_math.py
```
