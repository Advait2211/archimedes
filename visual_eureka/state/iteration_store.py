import json
import os
import sqlite3
from datetime import datetime, timezone


class IterationStore:
    def __init__(self, db_path: str = "state/eureka_runs.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                reward_code TEXT,
                candidate_codes TEXT,
                filter_stats TEXT,
                training_stats TEXT,
                eval_gif_path TEXT,
                eval_gif_paths TEXT,
                thumb_path TEXT,
                visual_critique TEXT,
                text_critique TEXT,
                user_critique TEXT,
                mean_reward REAL,
                success_rate REAL,
                model_path TEXT,
                timestamp TEXT,
                notes TEXT,
                UNIQUE(run_id, iteration)
            )
        """)
        for col, typedef in [("eval_gif_paths", "TEXT"), ("thumb_path", "TEXT")]:
            try:
                self.conn.execute(f"ALTER TABLE iterations ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    def save_iteration(self, run_id: str, iteration: int, **fields) -> int:
        timestamp = fields.pop("timestamp", datetime.now(timezone.utc).isoformat())

        json_fields = {}
        for key in ("candidate_codes", "filter_stats", "training_stats"):
            if key in fields and not isinstance(fields[key], str):
                json_fields[key] = json.dumps(fields[key])
            elif key in fields:
                json_fields[key] = fields[key]

        merged = {**fields, **json_fields}

        columns = ["run_id", "iteration", "timestamp"]
        values = [run_id, iteration, timestamp]

        valid_columns = {
            "reward_code", "candidate_codes", "filter_stats",
            "training_stats", "eval_gif_path", "eval_gif_paths", "thumb_path",
            "visual_critique", "text_critique", "user_critique", "mean_reward",
            "success_rate", "model_path", "notes",
        }

        for col in valid_columns:
            if col in merged:
                columns.append(col)
                values.append(merged[col])

        placeholders = ", ".join(["?"] * len(values))
        col_str = ", ".join(columns)

        cursor = self.conn.execute(
            f"INSERT OR REPLACE INTO iterations ({col_str}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_iteration(self, run_id: str, iteration: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id = ? AND iteration = ?",
            (run_id, iteration),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def get_all_iterations(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_best_iteration(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id = ? ORDER BY mean_reward DESC LIMIT 1",
            (run_id,),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def list_runs(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT run_id FROM iterations ORDER BY timestamp DESC"
        ).fetchall()
        return [r["run_id"] for r in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("candidate_codes", "filter_stats", "training_stats", "eval_gif_paths"):
            if key in d and d[key] is not None:
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        self.conn.close()
