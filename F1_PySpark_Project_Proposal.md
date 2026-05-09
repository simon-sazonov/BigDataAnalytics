# F1 Race Strategy Intelligence Platform
## PySpark Big Data Group Project — NOVA IMS

**Client:** Fictional Formula 1 Constructor  
**Consultant:** BigDataCompany  
**Dataset:** Formula 1 World Championship 1950–2024 (Rohan Rao, Kaggle)  
**Due:** 5th June 2026, 23:59h

---

## Project Summary

BigDataCompany is engaged by a fictional F1 constructor to build a data platform that helps the team make smarter pre-race and in-race decisions using 74 years of historical F1 data. The platform demonstrates how PySpark can handle the scale, complexity, and speed of motorsport data — and why Spark is the right technology choice for a client operating at this data volume and velocity.

---

## Dataset

| Field | Detail |
|---|---|
| Source | [Kaggle — Rohan Rao](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) |
| Coverage | 1950–2024 (Ergast-derived) |
| Files | 14 CSVs (~50–60 MB unzipped) |
| Key tables | `circuits`, `races`, `drivers`, `constructors`, `results`, `qualifying`, `lap_times`, `pit_stops`, `driver_standings`, `constructor_standings` |

---

## Notebooks Overview

| # | Notebook | Spark Components | Theme |
|---|---|---|---|
| 1 | ETL & Data Engineering | RDDs, DataFrames, SparkSQL | Bronze → Silver → Gold pipeline |
| 2 | Exploratory Analysis & Insights | SparkSQL, DataFrames | Strategic insights for management |
| 3 | Machine Learning Pipeline | MLlib, Pipelines | Qualifying & race outcome prediction |
| 4 | Graph Analytics | GraphFrames | Constructor lineage & driver networks |
| 5 *(bonus)* | Structured Streaming | Structured Streaming | Live lap-time leaderboard simulation |

---

## Notebook 1 — ETL & Data Engineering

**Spark components:** RDDs · DataFrames · SparkSQL

### Goal
Ingest all 14 raw CSVs into a medallion architecture and produce clean, analytical-ready Gold tables.

### Tasks

**Bronze layer**
- Ingest all 14 CSVs with explicit schemas (never `inferSchema=True` — not big-data safe)
- Store as Parquet partitioned by relevant keys (e.g. `year`, `raceId`)

**Silver layer**
- Null handling: flag and impute or drop missing values in `results`, `qualifying`, `lap_times`
- Schema conformance: normalise time strings (e.g. `"1:32.456"`) to milliseconds as integers
- Join validation: ensure referential integrity across `raceId`, `driverId`, `constructorId`
- Add audit columns: `ingested_at`, `source_file`

**Gold layer**
- `fact_race_result` — one row per driver per race, with all result, qualifying, standing, and pit-stop attributes
- `dim_driver` — SCD-2 style with nationality, dob, career span
- `dim_constructor` — with nationality, era classification (pre-turbo, turbo, V10, V8, hybrid)
- `dim_circuit` — with country, lat/lng, altitude, surface type

**RDD demonstration**
- Parse `lap_times.csv` as a raw RDD
- Use `mapPartitions` + `reduceByKey` to compute each driver's fastest lap per race
- Re-implement the same logic using DataFrames + SparkSQL
- Explicitly note: DataFrame approach is preferred at scale because of Catalyst optimiser and Tungsten execution engine; RDD approach shown for pedagogical completeness

**Big-data safe notes to include**
- Use `broadcast()` for small dimension joins (< 10 MB tables like `circuits`, `status`)
- Avoid `.collect()` before aggregation
- Use `partitionBy("year")` on Parquet writes
- Avoid Python-native `sorted()` — use Spark's `.orderBy()` instead

---

## Notebook 2 — Exploratory Analysis & Insights

**Spark components:** SparkSQL · DataFrames · Window Functions

### Goal
Answer 6–8 compelling business questions using SparkSQL that will directly feed the management presentation.

### Suggested Business Questions

1. **Circuit profitability** — Which circuits give constructors the highest average points return across all eras?
2. **Grid-to-finish translation** — How does starting grid position translate to final finish position, broken down by circuit and era?
3. **Pit-stop strategy evolution** — How has average pit-stop count and duration evolved per decade? Which teams pioneered undercut strategies?
4. **Constructor dominance windows** — Which constructors dominated consecutive seasons, and how long do dominance cycles last?
5. **Home-circuit advantage** — Do drivers score significantly more points at their home Grand Prix?
6. **Wet vs dry performance** — Which constructors perform above their season average at historically wet circuits (Spa, Interlagos, Suzuka)?
7. **Fastest lap trends** — How has fastest lap time per circuit evolved over the hybrid era vs V8/V10 eras?
8. **Championship decided by reliability** — In how many seasons did mechanical failures (DNFs from status) cost the eventual runner-up the title?

### Technical requirements
- All queries written as SparkSQL (`spark.sql(...)`) with registered temp views
- At least 3 queries use window functions (`RANK()`, `LAG()`, `LEAD()`, `ROW_NUMBER()`)
- Visualise results with `matplotlib` or `plotly` (convert to Pandas only after aggregation — flag this explicitly)

---

## Notebook 3 — Machine Learning Pipeline

**Spark components:** MLlib · Pipelines · CrossValidator · ParamGridBuilder

### Goal
Build two classification models to help the team predict race outcomes before and after qualifying.

---

### Model 1 — Qualifying Top-10 Predictor

**Business question:** Will a given driver make it into Q3 (top 10 in qualifying)?

**Target variable:** Binary — `in_q3` (1 if qualifying position ≤ 10, else 0)

**Features:**

| Feature | Source | Notes |
|---|---|---|
| Constructor encoded | `constructors` | StringIndexer → OneHotEncoder |
| Circuit encoded | `circuits` | StringIndexer → OneHotEncoder |
| Driver historical Q3 rate | `qualifying` | Rolling average over prior 5 races |
| Season year | `races` | Normalised |
| Driver points form | `driver_standings` | Points in prior 3 races |
| Constructor points form | `constructor_standings` | Same |

**Pipeline:**
```
StringIndexer → OneHotEncoder → VectorAssembler → LogisticRegression (baseline)
StringIndexer → OneHotEncoder → VectorAssembler → RandomForestClassifier (main)
```

**Evaluation:** ROC-AUC, Precision-Recall curve, Confusion Matrix  
**Validation:** `CrossValidator` with 5-fold CV and `ParamGridBuilder`  
**Holdout:** 2022–2024 seasons withheld from training

---

### Model 2 — Race Podium Predictor

**Business question:** Will a given driver finish on the podium (top 3)?

**Target variable:** Binary — `is_podium` (1 if `position` ≤ 3)

**Features:**

| Feature | Source | Notes |
|---|---|---|
| Grid position | `results` | Direct |
| Qualifying gap to pole (ms) | `qualifying` | q1/q2/q3 delta to fastest |
| Constructor podium rate at circuit | `results` + `races` | Historical proportion |
| Driver championship position | `driver_standings` | Before race weekend |
| Pit stop count (strategy proxy) | `pit_stops` | Count per race |
| Constructor encoded | `constructors` | OHE |
| Circuit encoded | `circuits` | OHE |

**Pipeline:**
```
StringIndexer → OneHotEncoder → VectorAssembler → GBTClassifier
```

**Evaluation:** ROC-AUC, Feature Importance chart, Calibration curve  
**Validation:** `CrossValidator` with 5-fold CV and `ParamGridBuilder` (tuning: maxDepth, maxIter, stepSize)  
**Holdout:** 2022–2024 seasons

---

### Big-data safe notes for ML
- Never convert full dataset to Pandas before training — only aggregated results
- Use `ml.Pipeline` for reproducibility and to avoid data leakage in CV
- Document that `GBTClassifier` is Spark-native and scales horizontally, unlike scikit-learn

---

## Notebook 4 — Graph Analytics

**Spark components:** GraphFrames · PageRank · ConnectedComponents · BFS · Motif Finding

### Goal
Model the F1 ecosystem as a graph to reveal hidden structural patterns in constructor evolution and driver-team relationships.

---

### Graph 1 — Constructor Lineage DAG

**Nodes:** Every unique constructor (≈ 210 in the dataset)  
**Edges:** Directed rebranding/takeover relationships  
- Source: F1DB `constructor_chronology` table joined back to Ergast `constructors`  
- Example edges: Tyrrell → BAR → Honda → Brawn → Mercedes

**Schema:**
```
vertices: constructorId, name, nationality, first_season, last_season, total_wins
edges: src (constructorId), dst (constructorId), relationship (rebranding / acquisition / split)
```

**Algorithms:**

| Algorithm | Output | Business meaning |
|---|---|---|
| `pageRank(resetProb=0.15, maxIter=10)` | Rank score per constructor | Which lineages are most "influential" in F1 history? |
| `connectedComponents()` | Cluster ID | Which teams are isolated vs part of a dynasty? |
| BFS shortest path | Path length | How many steps connect Williams to Mercedes? |
| Motif `(a)-[e]->(b); (b)-[f]->(c)` | Three-hop chains | Find all three-generation team lineages |

---

### Graph 2 — Driver–Constructor Bipartite Network

**Nodes:** All drivers + all constructors  
**Edges:** One edge per (driver, constructor, season) — weight = points scored that season

**Schema:**
```
vertices: id (driverId or constructorId), type (driver / constructor), name, nationality
edges: src (driverId), dst (constructorId), season, points, wins
```

**Algorithms:**

| Algorithm | Output | Business meaning |
|---|---|---|
| `pageRank()` | Career leverage score | Which constructors most amplified a driver's points? |
| `triangleCount()` | Triangle count per node | Which drivers were central connectors across teams? |
| Motif `(d)-[]->(c1); (d)-[]->(c2)` | Shared drivers | Which constructor pairs competed for the same drivers most often? |

---

### Big-data safe notes for GraphFrames
- Cache vertices and edges DataFrames before running iterative algorithms
- Set `maxIter` explicitly on PageRank (do not rely on convergence tolerance alone)
- Flag that GraphFrames requires a Spark checkpoint directory — set `spark.sparkContext.setCheckpointDir(...)`

---

## Notebook 5 (Bonus) — Structured Streaming

**Spark components:** Structured Streaming · withWatermark · Window Aggregations

### Goal
Simulate a live race engineer dashboard that tracks lap-by-lap position changes and gap-to-leader in real time.

### Architecture

```
lap_times.csv (sorted by raceId, lap)
        ↓
Python producer script
(copies per-lap CSV slices into /data/stream/ every 2 seconds)
        ↓
Spark readStream (file source, maxFilesPerTrigger=1)
        ↓
withWatermark("lap_timestamp", "10 seconds")
        ↓
window("lap_timestamp", "1 lap") + groupBy(driverId)
        ↓
Compute: position, gap_to_leader, stint_lap, pit_flag
        ↓
writeStream (outputMode="update", console sink)
```

### Key Spark Streaming features demonstrated

| Feature | Implementation |
|---|---|
| File source with controlled ingestion | `readStream.option("maxFilesPerTrigger", 1)` |
| Late-data handling | `withWatermark("lap_timestamp", "10 seconds")` |
| Windowed aggregation | `groupBy(window(...), "driverId").agg(...)` |
| Output modes | `outputMode("update")` for leaderboard; `outputMode("append")` for audit log |
| Checkpointing | `option("checkpointLocation", "/tmp/f1_checkpoint")` |

### Business narrative for management
> "This is what a race engineer sees in real time. As each lap completes, the system updates the leaderboard, flags drivers entering their pit window based on stint length, and highlights who is gaining or losing ground — all processed at scale with Spark Structured Streaming."

---

## Management Presentation Structure (10 slides)

| Slide | Title | Content |
|---|---|---|
| 1 | Title slide | Project name, team, client name, date |
| 2 | The client challenge | Why F1 is a big data problem — the 4 V's (Volume: 74 years of laps; Velocity: real-time telemetry; Variety: results, telemetry, weather; Veracity: historical data quality issues) |
| 3 | Why Apache Spark? | Distributed processing, fault tolerance, unified batch + streaming, ML at scale — framed as business benefits not technical specs |
| 4 | Strategic insight 1 | Circuit profitability map — where should the client focus development resources? |
| 5 | Strategic insight 2 | Grid position vs podium rate — does qualifying matter more than race pace at certain circuits? |
| 6 | Prediction tool: qualifying | What is the model, what can it tell the team before race weekend, headline accuracy metric |
| 7 | Prediction tool: podium | Which factors matter most for a top-3 finish — feature importance in plain English |
| 8 | Constructor heritage graph | Visual of the lineage graph — "who are we, historically?" — PageRank ranking of constructor influence |
| 9 | Live race dashboard | Screenshot of the streaming leaderboard — "this is what race engineers could see in real time" |
| 10 | References | Dataset, libraries, academic sources |

> **Tone guide:** No model parameters, no algorithm names, no hyperparameters. Frame everything as: "Here is what this tells the client" and "Here is what the client can now do."

---

## Big-Data Safety Checklist

Include a comment in every notebook wherever any of the following apply:

- [ ] Used `.collect()` after aggregation, not before — note why
- [ ] Used `broadcast()` for dimension joins under 10 MB
- [ ] Used `partitionBy()` on all Parquet writes
- [ ] Used `spark.sql()` / `.orderBy()` instead of Python-native sort
- [ ] All ML transformations inside a `Pipeline` (no leakage risk)
- [ ] `inferSchema=False` on all CSV reads — explicit schema defined
- [ ] GraphFrames checkpoint directory configured before iterative algorithms
- [ ] Streaming watermark and checkpoint location set

---

## Suggested Group Task Split (4–5 people)

| Member | Primary notebook | Secondary |
|---|---|---|
| Member 1 | Notebook 1 — ETL | Streaming producer script |
| Member 2 | Notebook 2 — EDA & SQL | Presentation slides 4–5 |
| Member 3 | Notebook 3 — ML (Model 1) | Presentation slides 6 |
| Member 4 | Notebook 3 — ML (Model 2) | Presentation slides 7 |
| Member 5 *(if 5)* | Notebook 4 — GraphFrames + Notebook 5 — Streaming | Presentation slides 8–9 |

---

## Key Links

| Resource | URL |
|---|---|
| Dataset (Kaggle) | https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 |
| F1DB (supplementary) | https://github.com/f1db/f1db |
| Jolpica-F1 API (Ergast successor) | https://github.com/jolpica/jolpica-f1 |
| FastF1 (telemetry, optional) | https://github.com/theOehrly/Fast-F1 |
| OpenF1 API (streaming source, optional) | https://openf1.org/ |
| GraphFrames docs | https://graphframes.github.io/graphframes/ |
| PySpark MLlib guide | https://spark.apache.org/docs/latest/ml-guide.html |
| Spark Structured Streaming guide | https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html |
