baseline
===
| SVM No. | Configurations |
| ------ | -------------- |
| 1 | Linear kernel, `C`=1, ovo, max 300k iter |
| 2 | rbf kernel, `C`=1, `gamma`=scale, ovo, max 300k iter |
| 3 | polynomial kernel (`degree`=3), `C`=1, `gamma`=auto, ovo, max 300k iter |

Results
---
| SVM No. | Mean acc | LDA acc |
| ------- | -------- | ------- |
| 1 | 0.5708 | 0.4753 |
| 2 | 0.6123 | 0.4458 |
| 3 | 0.5744 | 0.4867 |