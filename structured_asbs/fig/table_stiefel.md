| beta | reference | R-ASBS rerun | R-ASBS err | ours | our err |
|-----:|----------:|-------------:|-----------:|-----:|--------:|
| 0.001 | 7.9965 | 7.9546 | -0.0419 | 7.9958 | -0.0007 |
| 0.01 | 7.9628 | 7.9407 | -0.0221 | 7.9679 | +0.0051 |
| 0.1 | 7.6671 | 7.7262 | +0.0592 | 7.6893 | +0.0222 |
| 0.5 | 6.4171 | 6.5426 | +0.1256 | 6.5152 | +0.0981 |
| 1.3 | 4.7503 | 5.2008 | +0.4505 | 4.9016 | +0.1513 |
| 2 | 4.0976 | 4.6444 | +0.5469 | 4.2335 | +0.1359 |
| 5 | 3.4104 | 3.8893 | +0.4790 | 3.5212 | +0.1108 |
| 7 | 3.2905 | 3.6104 | +0.3199 | 3.3995 | +0.1090 |
| 10 | 3.2025 | 3.5275 | +0.3249 | 3.3102 | +0.1076 |
| 20 | 3.1005 | 3.3858 | +0.2853 | 3.2170 | +0.1164 |
| 50 | 3.0418 | 3.3116 | +0.2698 | 7.4330 * | +4.3912 * |
| 100 | 3.0279 | 3.2614 | +0.2335 | 8.3180 * | +5.2901 * |
| 200 | 3.0000 (exact) | 3.2482 | +0.2482 | - | - |
| 1000 | 3.0000 (exact) | 3.2086 | +0.2086 | - | - |
| 10000 | 3.0000 (exact) | 3.1873 | +0.1873 | - | - |
| 1e+06 | 3.0000 (exact) | 3.1847 | +0.1847 | - | - |

`*` training did not converge at this beta. The regression loss stays at 7e4 / 3e5 for the whole run, but that scale is a symptom, not the cause: a rerun with scale-free labels, whose normalised loss is O(1) at every beta, gives the same answer (+4.463 / +5.271). The cause is the 199-step Euler grid, unstable at an O(beta) score, together with the on-policy collection that samples from it. Reported as measured. R-ASBS is better than us at these two points on this grid. Refining the grid removes our error and not theirs -- see the second table below -- but that is a more expensive run and is reported separately rather than substituted in here.


### High beta, refined integration grid

| beta | method | steps | E | err | KS(E) |
|-----:|--------|------:|--:|----:|------:|
| 50 | ours | 398 | 3.1118 | +0.0685 | 0.479 |
| 50 | ours | 796 | 3.0656 | +0.0223 | 0.228 |
| 50 | ours | 1592 | 3.0534 | +0.0101 | 0.118 |
| 50 | R-ASBS | 199 | 3.3139 | +0.2721 | - |
| 50 | R-ASBS | 512 | 3.2969 | +0.2552 | - |
| 50 | R-ASBS | 1024 | 3.2897 | +0.2480 | - |
| 100 | ours | 796 | 6.5005 | +3.4695 | 1.000 |
| 100 | ours | 1592 | 3.7768 | +0.7458 | 1.000 |
| 100 | ours | 3184 | 3.0771 | +0.0461 | 0.312 |
| 100 | R-ASBS | 199 | 3.2626 | +0.2347 | - |
| 100 | R-ASBS | 512 | 3.2456 | +0.2177 | - |
| 100 | R-ASBS | 1024 | 3.2487 | +0.2208 | - |

Our error falls with the step count; R-ASBS's does not, because theirs is the source-tilting bias plus the QR retraction and neither is a function of step size. The KS column is the caveat: at beta = 100 our refined mean is within +0.046 but KS = 0.312 and the energy spread is 16x too broad, so refinement fixes the first moment and not the law. rasbs_port.py reports no KS, hence the dashes.
