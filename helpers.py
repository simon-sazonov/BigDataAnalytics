# Reusable functions — chart any driver(s) by name
# prepare_driver_stints(spark, gold, names) builds the stint table for a list of one or more driver names;
# render_movements(pdf, names) draws the colored trails. score="race" uses race finishing H2H, 
# score="quali" uses qualifying H2H.

def prepare_driver_stints(spark, GOLD_PATH, names=None, ids=None, years=None, teams=None):
    """Every team stint for a list of 3-letter driver abbreviations, with
    qualifying and race head-to-head vs teammates.
    `abbrs`: list[str] of 3-letter codes (e.g. ["VER","HAM"]).
    `years`: optional list[int] to restrict the seasons read. -> pandas DF."""
    from pyspark.sql import functions as F
    drivers = spark.read.parquet(f"{GOLD_PATH}/dim_driver")
    cons    = spark.read.parquet(f"{GOLD_PATH}/dim_constructor")
    canon = (F.when(F.col("constructor_name") == "Renault",      F.lit("Alpine F1 Team"))
              .when(F.col("constructor_name") == "Racing Point", F.lit("Aston Martin"))
              .when(F.col("constructor_name") == "AlphaTauri",   F.lit("RB F1 Team"))
              .when(F.col("constructor_name") == "Alfa Romeo",   F.lit("Sauber"))
              .otherwise(F.col("constructor_name")))
    cons = cons.withColumn("team", canon)
    facts   = spark.read.parquet(f"{GOLD_PATH}/fact_race_result")
    if years:
        facts = facts.filter(F.col("year").isin(list(years)))

    family = F.element_at(F.split(F.col("driver_name"), " "), -1)
    abbr = (F.when(F.col("code").isNotNull() & (F.col("code") != "\\N"),
                   F.upper(F.substring(F.col("code"), 1, 3)))
             .otherwise(F.upper(F.substring(family, 1, 3))))
    drivers = drivers.withColumn("abbr", abbr)
    if ids:
        foc = drivers.filter(F.col("driverId").isin([int(i) for i in ids]))
    elif names:
        foc = drivers.filter(F.upper(F.col("driver_name")).isin([n.upper() for n in names]))
    else:
        foc = drivers          # no names given -> keep everyone; teams/years narrow it
    foc = foc.select("driverId", "driver_name", "abbr")

    # one row per (driver, year, team); first_round orders mid-season moves
    stints = (facts.join(F.broadcast(foc), "driverId")
        .join(cons.select("constructorId", "team"), "constructorId")
        .groupBy("driverId", "driver_name", "abbr", "year", "team")
        .agg(F.min("round").alias("first_round"), F.count("*").alias("races")))
    if teams:
        stints = stints.filter(F.col("team").isin(list(teams)))

    # qualifying head-to-head per (driver, year)
    qall = (facts.filter(F.col("qualifying_position").isNotNull())
            .select("year", "raceId", "constructorId", "driverId", "qualifying_position"))
    mq = qall.select("raceId", "constructorId", F.col("driverId").alias("mate_id"),
                     F.col("qualifying_position").alias("mate_q"))
    qh = (qall.join(mq, ["raceId", "constructorId"]).filter(F.col("driverId") != F.col("mate_id"))
          .groupBy("year", "driverId")
          .agg(F.sum(F.when(F.col("qualifying_position") < F.col("mate_q"), 1).otherwise(0)).alias("q_wins"),
               F.sum(F.when(F.col("qualifying_position") > F.col("mate_q"), 1).otherwise(0)).alias("q_losses")))

    # race head-to-head per (driver, year, team): finished ahead of best teammate?
    res = facts.select("year", "raceId", "constructorId", "driverId", "positionOrder")
    res_foc = res.join(F.broadcast(foc.select("driverId")), "driverId")
    mr = res.select("raceId", "constructorId", F.col("driverId").alias("mate_id"),
                    F.col("positionOrder").alias("mate_po"))
    rh = (res_foc.join(mr, ["raceId", "constructorId"]).filter(F.col("driverId") != F.col("mate_id"))
          .groupBy("year", "constructorId", "driverId", "raceId", "positionOrder")
          .agg(F.min("mate_po").alias("best_mate"))
          .join(cons.select("constructorId", "team"), "constructorId")
          .groupBy("year", "driverId", "team")
          .agg(F.sum(F.when(F.col("positionOrder") < F.col("best_mate"), 1).otherwise(0)).alias("r_wins"),
               F.sum(F.when(F.col("positionOrder") > F.col("best_mate"), 1).otherwise(0)).alias("r_losses")))

    stints = (stints.join(qh, ["year", "driverId"], "left").fillna(0, ["q_wins", "q_losses"])
                    .join(rh, ["year", "driverId", "team"], "left").fillna(0, ["r_wins", "r_losses"]))
    return stints.toPandas()


def render_movements(pdf_stints, names=None, ids=None, years=None, out_png="movements.png",
                     score="race", min_races=1, legend=False, teams=None,
                     col_w=150.0, row_h=51.0, header_fontsize=16):
    """Draw colored movement trails, one per DRIVER (keyed on driverId, so even
    drivers that share a full name or 3-letter code stay separate). Nodes are
    labelled with the 3-letter abbreviation.
    Select drivers with either `names` (full names, case-insensitive) OR `ids`
    (exact driverId values). Omit both to draw every driver in pdf_stints.
    years = optional list[int] of seasons; score = "race" or "quali"."""
    import subprocess, collections
    from IPython.display import Image, display
    PALETTE = ["#1F5FD0", "#D4341E", "#1C9E4B", "#9A3FC0", "#E08A00",
               "#0E8E8E", "#B0306A", "#555555", "#2B7A0B", "#8A5A00"]
    wkey, lkey = ("r_wins", "r_losses") if score == "race" else ("q_wins", "q_losses")
    yset = set(years) if years else None
    tset = set(teams) if teams else None

    # map driverId -> (name, abbr) and a name lookup for selection
    id_name, id_abbr = {}, {}
    name_to_ids = collections.defaultdict(list)
    for r in pdf_stints.itertuples():
        did = int(r.driverId)
        id_name[did] = r.driver_name; id_abbr[did] = r.abbr
        if did not in name_to_ids[r.driver_name.lower()]:
            name_to_ids[r.driver_name.lower()].append(did)

    if ids:                                            # exact driverIds win
        order = [int(i) for i in ids if int(i) in id_name]
    elif names:                                        # full names, case-insensitive
        order = []
        for nm in names:
            for did in name_to_ids.get(nm.lower(), []):
                if did not in order: order.append(did)
    else:                                              # everyone, first-seen order
        order = list(dict.fromkeys(int(r.driverId) for r in pdf_stints.itertuples()))
    if not order:
        raise ValueError("none of the requested drivers are in the data")
    color = {did: PALETTE[i % len(PALETTE)] for i, did in enumerate(order)}

    seq = collections.defaultdict(list)
    for r in pdf_stints.itertuples():
        did = int(r.driverId)
        if did not in color:
            continue
        if yset and int(r.year) not in yset:
            continue
        if int(r.races) < min_races:
            continue
        if tset and r.team not in tset:
            continue
        seq[did].append((int(r.year), int(r.first_round), r.team,
                         int(getattr(r, wkey)), int(getattr(r, lkey))))
    for k in seq: seq[k].sort()

    present = []
    for did in order:
        for (y, fr, t, w, l) in seq.get(did, []):
            if t not in present: present.append(t)
    teams = [t for t in teams if t in present] if teams else present
    yrs = sorted({y for did in seq for (y, fr, t, w, l) in seq[did]})
    if not yrs:
        raise ValueError("no stints found for the requested drivers")
    colx = {t: i for i, t in enumerate(teams)}
    rowy = {y: i for i, y in enumerate(yrs)}
    nid = lambda did, y, t: "n_%d_%d_%d" % (order.index(did), y, colx[t])

    # fan out drivers sharing the same (year, team) cell
    cell_members = collections.defaultdict(list)
    for did in order:
        for (y, fr, t, w, l) in seq.get(did, []):
            if did not in cell_members[(y, t)]:
                cell_members[(y, t)].append(did)
    sub_w = 34.0
    node_x = {}
    for (y, t), mem in cell_members.items():
        n = len(mem)
        for j, did in enumerate(mem):
            node_x[(did, y, t)] = colx[t] * col_w + (j - (n - 1) / 2.0) * sub_w

    dot = ['digraph M {', '  graph [bgcolor="#FFFFFF"];',
           '  node [shape=box, style="filled,rounded", fontname="Helvetica-Bold", '
           'fontsize=11, fixedsize=true, width=0.7, height=0.42];',
           '  edge [penwidth=2.4, arrowsize=0.6];']
    for t in teams:
        dot.append(f'  h_{colx[t]} [pos="{colx[t]*col_w:.1f},{row_h*0.9:.1f}", label="{t}", '
                   f'shape=box, style=filled, fillcolor="#E8E8E8", fontcolor="#222222", '
                   f'fontsize={header_fontsize}, width=1.95, height=0.6];')
    for y in yrs:
        dot.append(f'  y_{y} [pos="{-col_w*0.8:.1f},{-rowy[y]*row_h:.1f}", label="{y}", '
                   f'shape=plaintext, fontsize=13, fontcolor="#333333", fixedsize=false];')
    for did in order:
        col = color[did]; ab = id_abbr.get(did, "")
        for (y, fr, t, w, l) in seq.get(did, []):
            sc = f'<BR/><FONT POINT-SIZE="8">{w}-{l}</FONT>' if (w + l) > 0 else ''
            dot.append(f'  {nid(did,y,t)} [pos="{node_x[(did,y,t)]:.1f},{-rowy[y]*row_h:.1f}", '
                       f'label=<{ab}{sc}>, fillcolor="{col}", fontcolor="#FFFFFF"];')
    for did in order:
        s = seq.get(did, []); col = color[did]
        for (y1, f1, t1, w1, l1), (y2, f2, t2, w2, l2) in zip(s, s[1:]):
            style = "dotted" if (y2 - y1) > 1 else "solid"
            extra = ", penwidth=3.0" if style == "dotted" else ""
            dot.append(f'  {nid(did,y1,t1)} -> {nid(did,y2,t2)} [color="{col}", style={style}{extra}];')
    if legend:
        ly = -(len(yrs) + 1) * row_h * 0.55
        for i, did in enumerate(order):
            dot.append(f'  lg_{i} [pos="{0.6*col_w:.1f},{ly - i*46:.1f}", '
                       f'label="{id_abbr.get(did,"")}  {id_name.get(did,"")}", '
                       f'shape=box, style=filled, fillcolor="{color[did]}", fontcolor="#FFFFFF", '
                       f'fontsize=11, width=2.6, height=0.4];')
        dot.append(f'  lg_note [pos="{0.6*col_w:.1f},{ly - len(order)*46:.1f}", '
                   f'label="solid = next stint / season   dotted = return after a gap year   '
                   f'(score = {score} H2H vs teammate)", '
                   f'shape=plaintext, fontsize=10, fontcolor="#333333", fixedsize=false];')

    dot.append('}')
    open(out_png + ".dot", "w").write("\n".join(dot))
    subprocess.run(["neato", "-n2", "-Tpng", "-Gdpi=150", out_png + ".dot", "-o", out_png], check=True)
    display(Image(out_png))
    return out_png

# Cluster statistics
# A cluster is a set of drivers (given by 3-letter abbreviations). These functions summarise the cluster across the teams 
# its members raced for, with totals and per-start rates so teams of different size/era are comparable. Omit abbrs to 
# include every driver for the chosen teams/years.

# ===== Section 7: cluster statistics (cluster = list of 3-letter abbreviations) =====
def _cluster_base(spark, GOLD_PATH, names=None, ids=None, years=None, teams=None):
    """Shared base: race results of the cluster's drivers with canonical team
    names. Returns (base, facts_team): `base` = cluster rows; `facts_team` = all
    drivers (needed for teammate comparisons)."""
    from pyspark.sql import functions as F
    drivers = spark.read.parquet(f"{GOLD_PATH}/dim_driver")
    cons    = spark.read.parquet(f"{GOLD_PATH}/dim_constructor")
    facts   = spark.read.parquet(f"{GOLD_PATH}/fact_race_result")
    if years:
        facts = facts.filter(F.col("year").isin(list(years)))
    canon = (F.when(F.col("constructor_name") == "Renault",      F.lit("Alpine F1 Team"))
              .when(F.col("constructor_name") == "Racing Point", F.lit("Aston Martin"))
              .when(F.col("constructor_name") == "AlphaTauri",   F.lit("RB F1 Team"))
              .when(F.col("constructor_name") == "Alfa Romeo",   F.lit("Sauber"))
              .otherwise(F.col("constructor_name")))
    cons = cons.withColumn("team", canon)
    family = F.element_at(F.split(F.col("driver_name"), " "), -1)
    abbr = (F.when(F.col("code").isNotNull() & (F.col("code") != "\\N"),
                   F.upper(F.substring(F.col("code"), 1, 3)))
             .otherwise(F.upper(F.substring(family, 1, 3))))
    drivers = drivers.withColumn("abbr", abbr)
    if ids:
        foc = drivers.filter(F.col("driverId").isin([int(i) for i in ids]))
    elif names:
        foc = drivers.filter(F.upper(F.col("driver_name")).isin([n.upper() for n in names]))
    else:
        foc = drivers
    foc = foc.select("driverId", "driver_name", "abbr")
    facts_team = facts.join(cons.select("constructorId", "team"), "constructorId")
    base = facts_team.join(F.broadcast(foc), "driverId")
    if teams:
        base = base.filter(F.col("team").isin(list(teams)))
    return base, facts_team


def cluster_team_summary(spark, GOLD_PATH, names=None, ids=None, years=None, teams=None, add_totals=True):
    """One comparable row per team the cluster raced for. Totals AND per-start
    rates so teams of different size/era can be compared. -> pandas DF."""
    from pyspark.sql import functions as F
    base, facts_team = _cluster_base(spark, GOLD_PATH, names, ids, years, teams)

    # teammate qualifying H2H (vs same-constructor cars, any driver)
    mq = facts_team.filter(F.col("qualifying_position").isNotNull()).select(
            "raceId", "constructorId", F.col("driverId").alias("mid"),
            F.col("qualifying_position").alias("mqp"))
    qj = (base.filter(F.col("qualifying_position").isNotNull())
              .join(mq, ["raceId", "constructorId"]).filter(F.col("driverId") != F.col("mid")))
    qh = qj.groupBy("team").agg(
        F.sum(F.when(F.col("qualifying_position") < F.col("mqp"), 1).otherwise(0)).alias("q_wins"),
        F.sum(F.when(F.col("qualifying_position") > F.col("mqp"), 1).otherwise(0)).alias("q_losses"))

    # teammate race H2H (finished ahead of best teammate)
    mr = facts_team.select("raceId", "constructorId", F.col("driverId").alias("mid"),
                           F.col("positionOrder").alias("mpo"))
    rj = (base.join(mr, ["raceId", "constructorId"]).filter(F.col("driverId") != F.col("mid"))
              .groupBy("team", "raceId", "driverId", "positionOrder")
              .agg(F.min("mpo").alias("best")))
    rh = rj.groupBy("team").agg(
        F.sum(F.when(F.col("positionOrder") < F.col("best"), 1).otherwise(0)).alias("r_wins"),
        F.sum(F.when(F.col("positionOrder") > F.col("best"), 1).otherwise(0)).alias("r_losses"))

    agg = base.groupBy("team").agg(
        F.countDistinct("driverId").alias("drivers"),
        F.countDistinct(F.concat_ws("_", "year", "driverId")).alias("driver_seasons"),
        F.count("*").alias("starts"),
        F.sum(F.when(F.col("positionOrder") == 1, 1).otherwise(0)).alias("wins"),
        F.sum(F.when(F.col("positionOrder") <= 3, 1).otherwise(0)).alias("podiums"),
        F.sum(F.coalesce(F.col("points"), F.lit(0))).alias("points"),
        F.avg("positionOrder").alias("avg_finish"),
        F.avg("qualifying_position").alias("avg_grid"))

    out = (agg.join(qh, "team", "left").join(rh, "team", "left")
              .fillna(0, ["q_wins", "q_losses", "r_wins", "r_losses"])
              .withColumn("win_pct",          F.round(100 * F.col("wins")    / F.col("starts"), 2))
              .withColumn("podium_pct",       F.round(100 * F.col("podiums") / F.col("starts"), 2))
              .withColumn("points_per_start", F.round(F.col("points") / F.col("starts"), 3))
              .withColumn("avg_finish",       F.round("avg_finish", 2))
              .withColumn("avg_grid",         F.round("avg_grid", 2))
              .withColumn("quali_h2h", F.concat_ws("-", "q_wins", "q_losses"))
              .withColumn("race_h2h",  F.concat_ws("-", "r_wins", "r_losses")))
    cols = ["team", "drivers", "driver_seasons", "starts", "wins", "podiums", "points",
            "win_pct", "podium_pct", "points_per_start", "avg_finish", "avg_grid",
            "quali_h2h", "race_h2h"]
    pdf = out.select(cols).orderBy(F.desc("points_per_start")).toPandas()

    if add_totals:
        # cluster-wide totals (distinct drivers, not a sum across teams)
        ov = base.agg(
            F.countDistinct("driverId").alias("drivers"),
            F.countDistinct(F.concat_ws("_", "year", "driverId")).alias("driver_seasons"),
            F.count("*").alias("starts"),
            F.sum(F.when(F.col("positionOrder") == 1, 1).otherwise(0)).alias("wins"),
            F.sum(F.when(F.col("positionOrder") <= 3, 1).otherwise(0)).alias("podiums"),
            F.sum(F.coalesce(F.col("points"), F.lit(0))).alias("points"),
            F.avg("positionOrder").alias("avg_finish"),
            F.avg("qualifying_position").alias("avg_grid")).collect()[0]
        qov = qj.agg(
            F.sum(F.when(F.col("qualifying_position") < F.col("mqp"), 1).otherwise(0)).alias("w"),
            F.sum(F.when(F.col("qualifying_position") > F.col("mqp"), 1).otherwise(0)).alias("l")).collect()[0]
        rov = rj.agg(
            F.sum(F.when(F.col("positionOrder") < F.col("best"), 1).otherwise(0)).alias("w"),
            F.sum(F.when(F.col("positionOrder") > F.col("best"), 1).otherwise(0)).alias("l")).collect()[0]
        import pandas as pd
        st = pdf["starts"].sum()                       # total cluster starts
        def wavg(col, nd):                             # starts-weighted mean of team rows
            return round((pdf[col] * pdf["starts"]).sum() / st, nd) if st else 0.0
        total = {
            "team": "ALL (cluster)",
            "drivers": ov["drivers"], "driver_seasons": ov["driver_seasons"],
            "starts": int(st),
            "wins": int(pdf["wins"].sum()), "podiums": int(pdf["podiums"].sum()),
            "points": pdf["points"].sum(),
            "win_pct":          wavg("win_pct", 2),          # = Σ(win_pct_i * starts_i)/Σstarts
            "podium_pct":       wavg("podium_pct", 2),
            "points_per_start": wavg("points_per_start", 3),
            "avg_finish":       wavg("avg_finish", 2),
            "avg_grid":         wavg("avg_grid", 2),
            "quali_h2h": f"{qov['w']}-{qov['l']}",
            "race_h2h":  f"{rov['w']}-{rov['l']}"}
        pdf = pd.concat([pdf, pd.DataFrame([total])], ignore_index=True)
    return pdf


def cluster_bar(summary_pdf, metric="points_per_start", out_png="cluster_bar.png"):
    """Horizontal bar chart of one summary metric per team (sorted)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    d = summary_pdf.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.45 * len(d))))
    ax.barh(d["team"].astype(str), d[metric], color="#1F5FD0")
    ax.set_xlabel(metric)
    ax.set_title(f"Cluster — {metric} by team")
    for i, v in enumerate(d[metric]):
        ax.text(v, i, f" {v:g}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
    display(Image(out_png)); return out_png


def cluster_heatmap(spark, GOLD_PATH, names=None, ids=None, years=None, teams=None,
                    metric="drivers", out_png="cluster_heatmap.png"):
    """Heatmap of teams (rows) x years (cols) for the cluster.
    metric = "drivers" (distinct cluster drivers), "points_per_start", or "wins"."""
    from pyspark.sql import functions as F
    import numpy as np
    base, _ = _cluster_base(spark, GOLD_PATH, names, ids, years, teams)
    if metric == "drivers":
        g = base.groupBy("team", "year").agg(F.countDistinct("driverId").alias("v"))
    elif metric == "points_per_start":
        g = base.groupBy("team", "year").agg(
            F.round(F.sum(F.coalesce(F.col("points"), F.lit(0))) / F.count("*"), 2).alias("v"))
    elif metric == "wins":
        g = base.groupBy("team", "year").agg(
            F.sum(F.when(F.col("positionOrder") == 1, 1).otherwise(0)).alias("v"))
    else:
        raise ValueError("metric must be 'drivers', 'points_per_start' or 'wins'")
    pdf = g.toPandas()
    mat = pdf.pivot(index="team", columns="year", values="v").sort_index()
    mat = mat.reindex(sorted(mat.columns), axis=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * mat.shape[1] + 2),
                                    max(3, 0.5 * mat.shape[0] + 1)))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:g}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Cluster {metric}: team x year")
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
    display(Image(out_png)); return out_png


# Combined: describe_cluster(...) with output flags
# Toggle each output: show_table, show_bar, show_heatmap. Pick which metric the bar (bar_metric) 
# and heatmap (heatmap_metric) use.

def describe_cluster(spark, GOLD_PATH, names=None, ids=None, years=None, teams=None,
                     show_flow=True, show_table=False, show_bar=False, show_heatmap=False,
                     bar_metric="points_per_start", heatmap_metric="drivers",
                     min_races=1, legend=False):
    """One call to describe a cluster. By default shows ONLY the flow chart;
    turn on the others with show_table / show_bar / show_heatmap.
    The table is displayed (not returned) to avoid showing it twice;
    use cluster_team_summary(...) directly if you need the DataFrame."""
    from IPython.display import display
    if show_flow:
        pdf = prepare_driver_stints(spark, GOLD_PATH, names, ids, years, teams)
        render_movements(pdf, names=names, ids=ids, years=years, teams=teams,
                         min_races=min_races, legend=legend)
    summary = None
    if show_table or show_bar:
        summary = cluster_team_summary(spark, GOLD_PATH, names, ids, years, teams)
    if show_table:
        display(summary)
    if show_bar:
        cluster_bar(summary, metric=bar_metric)
    if show_heatmap:
        cluster_heatmap(spark, GOLD_PATH, names, ids, years, teams, metric=heatmap_metric)
    return None

