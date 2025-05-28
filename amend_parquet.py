import argparse, sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path, PurePosixPath

def sample_file(src: Path, dst: Path, frac: float, rng: np.random.Generator):
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(dst, schema, compression='zstd')
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        n = rg.num_rows
        k = int(n * frac)
        if k > 0:
            idx = rng.choice(n, k, replace=False)
            writer.write_table(rg.take(pa.array(idx)))
    writer.close()

def main(argv):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--src", default="data")
    p.add_argument("--dst", default="data_sampled")
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for f in sorted(src_dir.glob("*.parquet")):
        sample_file(f, dst_dir / f.name, args.frac, rng)

if __name__ == "__main__":
    main(sys.argv[1:])
