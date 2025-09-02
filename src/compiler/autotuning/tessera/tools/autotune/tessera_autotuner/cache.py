
import os, sqlite3, json, time
from typing import Optional, Dict, Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT,
  device_class TEXT,
  kernel_id TEXT,
  problem_sig TEXT,
  schedule_key TEXT,
  config_json TEXT,
  budget_iters INTEGER,
  metrics_json TEXT,
  objective_value REAL
);
CREATE INDEX IF NOT EXISTS idx_lookup ON trials(device_class,kernel_id,problem_sig,schedule_key);
"""

class CacheDB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
    def close(self):
        self.conn.close()
    def put_trial(self, row: Dict[str,Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO trials (ts,device_class,kernel_id,problem_sig,schedule_key,config_json,budget_iters,metrics_json,objective_value) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row.get("ts"),
                row.get("device_class"),
                row.get("kernel_id"),
                json.dumps(row.get("problem_sig", {}), sort_keys=True),
                row.get("schedule_key"),
                json.dumps(row.get("config", {}), sort_keys=True),
                int(row.get("budget_iters") or 0),
                json.dumps(row.get("metrics", {}), sort_keys=True),
                float(row.get("objective_value") or 0.0),
            )
        )
        self.conn.commit()
    def lookup(self, device_class: str, kernel_id: str, problem_sig: dict, schedule_key: str):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT metrics_json FROM trials WHERE device_class=? AND kernel_id=? AND problem_sig=? AND schedule_key=? ORDER BY id DESC LIMIT 1",
            (device_class, kernel_id, json.dumps(problem_sig, sort_keys=True), schedule_key)
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None
