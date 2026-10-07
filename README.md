# Surface Aware Optical Control

This public repository accompanies our academic paper on the **Surface Aware methodology**. It contains the analysis program and the test data used to produce the results reported in the paper.

Robotics and AI Research Labs, Huai'an University
By: Dr. Amir Ali Mokhtarzadeh

## Purpose and scope

This repository exists only for our research on the Surface Aware methodology, as described in our academic paper.

- **Research only.** The material is published purely for research investigation and for the reproducibility of the results in the paper.
- **No industrial or production value.** The code and data are not a product, are not intended for industrial or production use, and come with no warranty or support.
- **No changes or updates allowed.** The contents are kept as they were used for the paper. Modifications, updates and contributions are not accepted.

## Contents

| File | Description |
| --- | --- |
| `surface_aware_optic_test_v4.py` | Analysis program. Reads a GapCal test log and calculates the statistics, tables and figures. It does not generate or modify measurements. |
| `Gapcal_Test1.csv` | Test log recorded with GapCal, a contactless gap and flush measurement device developed in our lab. |

## Reproducing the results

The program needs Python 3 with `numpy`, `pandas` and `matplotlib` (plus `openpyxl` if the input is an Excel file).

```bash
python surface_aware_optic_test_v4.py Gapcal_Test1.csv
python surface_aware_optic_test_v4.py Gapcal_Test1.csv --output results
```

The result tables (CSV) and figures (PNG) are written to the output directory, `results` by default.

The definitions of the calculated quantities (error, MAE, RMSE, bias, sigma and the in-tolerance criterion) are stated in the header of the program and in the paper.
