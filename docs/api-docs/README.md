<!-- markdownlint-disable -->

# API Overview

## Modules

- [`great`](./great.md#module-great)
- [`great_dataset`](./great_dataset.md#module-great_dataset)
- [`great_start`](./great_start.md#module-great_start)
- [`great_trainer`](./great_trainer.md#module-great_trainer)
- [`great_utils`](./great_utils.md#module-great_utils)
- [`metrics`](./metrics.md#module-metrics)

## Classes

- [`great.GReaT`](./great.md#class-great): GReaT Class
- [`great_dataset.GReaTDataCollator`](./great_dataset.md#class-greatdatacollator): GReaT Data Collator
- [`great_dataset.GReaTDataset`](./great_dataset.md#class-greatdataset): GReaT Dataset
- [`great_start.CategoricalStart`](./great_start.md#class-categoricalstart): Categorical Starting Feature
- [`great_start.ContinuousStart`](./great_start.md#class-continuousstart): Continuous Starting Feature
- [`great_start.GReaTStart`](./great_start.md#class-greatstart): Abstract super class GReaT Start
- [`great_start.RandomStart`](./great_start.md#class-randomstart): Random Starting Features
- [`great_trainer.GReaTTrainer`](./great_trainer.md#class-greattrainer): GReaT Trainer

### Metrics

- [`metrics.BaseMetric`](./metrics.md#class-basemetric): Abstract base class for all metrics
- [`metrics.ColumnShapes`](./metrics.md#class-columnshapes): Per-column distribution similarity
- [`metrics.ColumnPairTrends`](./metrics.md#class-columnpairtrends): Pairwise correlation preservation
- [`metrics.BasicStatistics`](./metrics.md#class-basicstatistics): Summary statistics comparison
- [`metrics.DiscriminatorMetric`](./metrics.md#class-discriminatormetric): Real vs synthetic classifier
- [`metrics.MLEfficiency`](./metrics.md#class-mlefficiency): Train on synthetic, test on real
- [`metrics.DistanceToClosestRecord`](./metrics.md#class-distancetoclosestrecord): Distance to nearest real record
- [`metrics.kAnonymization`](./metrics.md#class-kanonymization): k-Anonymity evaluation
- [`metrics.lDiversity`](./metrics.md#class-ldiversity): Sensitive attribute diversity
- [`metrics.IdentifiabilityScore`](./metrics.md#class-identifiabilityscore): Re-identification risk
- [`metrics.DeltaPresence`](./metrics.md#class-deltapresence): Presence disclosure risk
- [`metrics.MembershipInference`](./metrics.md#class-membershipinference): Membership inference attack risk


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
