
import sqlite3, json, argparse, os

SCHEMA = r"""
CREATE TABLE IF NOT EXISTS schedules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  arch TEXT NOT NULL,
  dtype TEXT NOT NULL,
  op TEXT NOT NULL,
  shape TEXT NOT NULL,
  knobs TEXT NOT NULL,
  result_ms REAL NOT NULL,
  tstamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sched ON schedules(arch, dtype, op);
"""

def ensure_db(path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  conn = sqlite3.connect(path)
  conn.executescript(SCHEMA)
  conn.commit()
  return conn

def insert_row(conn, arch, dtype, op, shape, knobs, result_ms):
  conn.execute(
    "INSERT INTO schedules(arch,dtype,op,shape,knobs,result_ms) VALUES(?,?,?,?,?,?)",
    (arch,dtype,op,shape,json.dumps(knobs),result_ms)
  )
  conn.commit()

def best_row(conn, arch, dtype, op, shape):
  cur = conn.execute(
    "SELECT knobs, result_ms FROM schedules WHERE arch=? AND dtype=? AND op=? AND shape=? ORDER BY result_ms ASC LIMIT 1",
    (arch,dtype,op,shape)
  )
  row = cur.fetchone()
  if not row: return None
  return {"knobs": json.loads(row[0]), "result_ms": row[1]}

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--db", required=True)
  ap.add_argument("--insert", action="store_true")
  ap.add_argument("--best", action="store_true")
  ap.add_argument("--arch")
  ap.add_argument("--dtype")
  ap.add_argument("--op")
  ap.add_argument("--shape")
  ap.add_argument("--knobs_json")
  ap.add_argument("--ms", type=float)
  args = ap.parse_args()

  conn = ensure_db(args.db)
  if args.insert:
    insert_row(conn, args.arch, args.dtype, args.op, args.shape, json.loads(args.knobs_json), args.ms)
    print("OK")
  elif args.best:
    print(json.dumps(best_row(conn, args.arch, args.dtype, args.op, args.shape)))
