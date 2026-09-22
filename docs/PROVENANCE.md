# Provenance

This repository develops the Ecuador census reconstruction work originally assembled in the CS1605 Group 02 project by 于乐水、吕峻锋、薛飞扬、陈尚尧. It reorganizes that work into an independent codebase; individual contribution statements will be added separately.

The initial modeling implementation and tests derive from `CS1605/model` at commit `7dc10a8`. The final combined dataset, reference metrics and maps derive from the reviewed project delivery. Subsequent engineering changes fix ablation metric export, cache validation, local paths and reproducibility entry points. No original benchmark scores have been changed or relabeled as newly reproduced results.

Raw census data are provided by INEC Ecuador. Source URLs, checksums, citation information and recorded usage terms are retained in `data/metadata/data_sources.csv`; the executable download manifest is `configs/sources.toml`. The 2010 and 2022 census definitions differ, and harmonized indicators retain comparability limitations. Reference geometry and quality metadata are retained alongside the source table.

The original course report, presentation material, QGIS delivery project and development branches are not runtime dependencies and are not included here.
