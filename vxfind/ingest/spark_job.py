"""The write path, stage 1: from the raw card dump to catalog records. A Spark job.

    python -m vxfind.ingest.spark_job --snapshot old
    python -m vxfind.ingest.spark_job --snapshot new --previous old

Stages, each a step of one Spark plan:
  1. read      the five parquet files of one dump revision (every dataset on the Hub)
  2. select    the rows tagged as robot-learning datasets
  3. parse     card -> record, in Python workers over Arrow batches (vxfind.catalog.parse)
  4. enrich    join the API listing (access gating, revision); mark families of near-identical
               tables and choose one head per family
  5. validate  count what could not be parsed; nothing is dropped silently
  6. diff      against the records of a previous snapshot: created, changed, deleted
  7. write     records, changes and statistics under data/vxfind/<snapshot>/

It runs in local mode here (one machine, all cores). Nothing in it depends on that: the same
plan runs on a cluster by changing the master and the paths. Output is written by the driver
through Arrow, because a local Windows Spark cannot write files without Hadoop's native
helper; on a cluster stage 7 is a plain `df.write.parquet`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / "data" / "hub"
OUT = ROOT / "data" / "vxfind"
SCALARS = [("table_id", "string"), ("owner", "string"), ("name", "string"), ("url", "string"), ("created", "string"),
           ("last_modified", "string"), ("downloads", "long"), ("likes", "long"), ("licence", "string"),
           ("schema_status", "string"), ("codebase_version", "string"), ("robot_type", "string"), ("fps", "double"),
           ("total_episodes", "long"), ("total_frames", "long"), ("total_tasks", "long"), ("data_mb", "double"),
           ("video_mb", "double"), ("n_columns", "int"), ("n_cameras", "int"), ("action_dim", "long"), ("state_dim", "long"),
           ("has_description", "boolean"), ("is_scratch", "boolean"), ("has_stamp", "boolean"), ("card_chars", "long"),
           ("card_hash", "string"), ("schema_hash", "string"), ("content_hash", "string"), ("summary", "string"),
           ("description", "string")]
NESTED = ("tags", "arxiv", "columns", "cameras", "action_names", "name_tokens", "problems")
SCHEMA = ", ".join(f"{n} {t}" for n, t in SCALARS) + ", " + ", ".join(f"{n}_json string" for n in NESTED)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def manifest() -> dict[str, Any]:
    return json.loads((HUB / "MANIFEST.json").read_text(encoding="utf-8"))


def parse_batches(batches: Iterator[Any]) -> Iterator[Any]:
    """Runs in the Python workers: one pandas frame of dump rows in, one frame of records out."""
    import pandas as pd

    from vxfind.catalog.parse import parse_row
    for pdf in batches:
        rows = []
        for r in pdf.itertuples(index=False):
            tags = list(r.tags) if r.tags is not None else []
            rec = parse_row(r.datasetId, r.author, r.last_modified, r.createdAt, r.downloads, r.likes, tags, r.card)
            row = {n: rec[n] for n, _ in SCALARS}
            for n in NESTED:
                row[f"{n}_json"] = json.dumps(rec[n], ensure_ascii=False)
            rows.append(row)
        out = pd.DataFrame(rows, columns=[n for n, _ in SCALARS] + [f"{n}_json" for n in NESTED])
        for n, t in SCALARS:
            if t in ("long", "int"):
                out[n] = out[n].astype("Int64")
            elif t == "boolean":
                out[n] = out[n].astype("boolean")
        yield out


def session(app: str) -> Any:
    from pyspark.sql import SparkSession
    os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    spark = (SparkSession.builder.master(os.environ.get("VXFIND_SPARK_MASTER", "local[*]")).appName(app)
             .config("spark.ui.enabled", "false").config("spark.driver.memory", "8g")
             .config("spark.sql.shuffle.partitions", "16").config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
             .config("spark.python.worker.reuse", "true").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run(snapshot: str, previous: str | None = None, limit: int | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    man = manifest()
    snap = man["snapshots"][snapshot]
    folder = HUB / "dump" / snap["revision"][:12]
    out = OUT / snapshot
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spark = session(f"vxfind-ingest-{snapshot}")
    stats: dict[str, Any] = {"snapshot": snapshot, "dump_revision": snap["revision"], "dump_published": snap["published"],
                             "spark": spark.version, "master": spark.sparkContext.master}

    # 1-2. read, select
    # Files are named one by one. A directory or a glob would make Hadoop list the folder, which
    # on Windows needs its native helper; a list of files needs nothing.
    raw = spark.read.parquet(*[p.as_uri() for p in sorted(folder.glob("*.parquet"))])
    stats["rows_in_the_dump"] = raw.count()
    tagged = raw.where(F.array_contains("tags", "LeRobot")).select(
        "datasetId", "author", "last_modified", "createdAt", "downloads", "likes", "tags", "card")
    if limit:
        tagged = tagged.limit(limit)
    tagged = tagged.repartition(16)

    # 3. parse
    recs = tagged.mapInPandas(parse_batches, schema=SCHEMA)

    # 4. enrich: the listing knows what the dump does not (gating, revision)
    listing_file = HUB / man["listings"][-1]["file"]
    listing = spark.read.json(listing_file.as_uri()).select(
        F.col("id").alias("table_id"), F.col("gated").cast("string").alias("gated"), F.col("sha").alias("revision"),
        F.col("private"), F.col("disabled"))
    recs = recs.join(F.broadcast(listing), "table_id", "left")
    recs = recs.withColumn("gated", F.when(F.col("gated").isin("auto", "manual", "true"), F.col("gated")).otherwise(F.lit("no")))
    # Families: the same owner, the same schema, and a name that differs only in digits (takes,
    # dates, counters). One table per family is its head: the one with the most frames.
    stem = F.regexp_replace(F.lower("name"), r"\d+", "#")
    recs = recs.withColumn("family", F.sha2(F.concat_ws("|", "owner", stem, F.coalesce("schema_hash", F.lit("-"))), 256).substr(1, 16))
    w = Window.partitionBy("family")
    order = Window.partitionBy("family").orderBy(F.col("has_description").desc(), F.col("total_frames").desc_nulls_last(),
                                                 F.col("last_modified").desc(), F.col("table_id"))
    recs = (recs.withColumn("family_size", F.count("*").over(w)).withColumn("family_rank", F.row_number().over(order))
            .withColumn("is_head", F.col("family_rank") == 1).withColumn("created_month", F.substring("created", 1, 7)))
    recs = recs.cache()

    # 5. validate
    n = recs.count()
    agg = recs.agg(
        F.sum(F.when(F.col("schema_status") == "ok", 1).otherwise(0)).alias("schema_ok"),
        F.sum(F.when(F.col("schema_status") == "absent", 1).otherwise(0)).alias("schema_absent"),
        F.sum(F.when(F.col("schema_status") == "unparsable", 1).otherwise(0)).alias("schema_unparsable"),
        F.sum(F.col("has_description").cast("int")).alias("with_a_description"),
        F.sum(F.col("is_scratch").cast("int")).alias("scratch_named"),
        F.sum(F.col("is_head").cast("int")).alias("families"),
        F.sum(F.when(F.col("gated") != "no", 1).otherwise(0)).alias("gated"),
        F.countDistinct("owner").alias("owners"), F.countDistinct("robot_type").alias("robot_types"),
        F.sum("total_episodes").alias("episodes"), F.sum("total_frames").alias("frames"),
        F.sum(F.coalesce("data_mb", F.lit(0.0)) + F.coalesce("video_mb", F.lit(0.0))).alias("declared_mb"),
        F.sum(F.when(F.col("data_mb").isNotNull() | F.col("video_mb").isNotNull(), 1).otherwise(0)).alias("tables_declaring_size"),
        F.max("family_size").alias("largest_family")).collect()[0].asDict()
    stats.update({"tables": n, **{k: (float(v) if isinstance(v, float) else v) for k, v in agg.items()}})
    stats["created_per_month"] = {r["created_month"]: r["count"] for r in recs.groupBy("created_month").count().orderBy("created_month").collect()}
    stats["top_robot_types"] = {str(r["robot_type"]): r["count"] for r in recs.groupBy("robot_type").count().orderBy(F.desc("count")).limit(25).collect()}
    stats["codebase_versions"] = {str(r["codebase_version"]): r["count"] for r in recs.groupBy("codebase_version").count().orderBy(F.desc("count")).collect()}
    stats["family_sizes"] = {str(r["bucket"]): r["count"] for r in recs.where("is_head").select(
        F.when(F.col("family_size") == 1, "1").when(F.col("family_size") <= 5, "2-5").when(F.col("family_size") <= 50, "6-50")
        .otherwise("51+").alias("bucket")).groupBy("bucket").count().collect()}

    # 6. diff against a previous snapshot
    changes = None
    if previous:
        prev = spark.read.parquet((OUT / previous / "tables.parquet").as_uri()).select(
            F.col("table_id"), F.col("content_hash").alias("p_content"), F.col("schema_hash").alias("p_schema"),
            F.col("last_modified").alias("p_modified"))
        j = recs.select("table_id", "content_hash", "schema_hash", "last_modified").join(prev, "table_id", "full_outer")
        changes = j.select(
            "table_id",
            F.when(F.col("p_content").isNull(), "created").when(F.col("content_hash").isNull(), "deleted")
            .when(F.col("content_hash") != F.col("p_content"), "changed").otherwise("same").alias("change"),
            (F.coalesce(F.col("schema_hash"), F.lit("-")) != F.coalesce(F.col("p_schema"), F.lit("-"))).alias("schema_changed"),
            F.col("p_modified"), F.col("last_modified")).where("change != 'same'").cache()
        by = {r["change"]: r["count"] for r in changes.groupBy("change").count().collect()}
        stats["changes_since_previous"] = {"previous": previous, **by,
                                           "changed_with_a_different_schema": changes.where("change = 'changed' AND schema_changed").count()}

    # 7. write (by the driver, through Arrow)
    pq.write_table(recs.toArrow(), out / "tables.parquet", compression="zstd")
    if changes is not None:
        pq.write_table(changes.toArrow(), out / "changes.parquet", compression="zstd")
    stats["seconds"] = round(time.time() - t0, 1)
    stats["output_mb"] = round((out / "tables.parquet").stat().st_size / 1e6, 1)
    (out / "ingest_stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    spark.stop()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Catalog ingestion (Spark).")
    ap.add_argument("--snapshot", required=True, choices=["old", "new"])
    ap.add_argument("--previous", choices=["old", "new"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    stats = run(a.snapshot, a.previous, a.limit)
    log(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
