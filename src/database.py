import sqlite3
import pandas as pd
import os
import shutil
import tempfile

# SQLite cannot reliably write to network/FUSE-mounted filesystems.
# We write to a temp file and copy to outputs/ for portability.
_TMP_DB = os.path.join(tempfile.gettempdir(), 'jay_portfolio_data.db')
DB_PATH = os.path.join('outputs', 'data.db')


def get_connection():
    """Return a connection to the local-disk SQLite database."""
    return sqlite3.connect(_TMP_DB)


def save_to_db(df, table_name, if_exists='replace'):
    """Persist a DataFrame as a SQLite table, then copy db to outputs/."""
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    # Copy db file to outputs/ folder so it ships with the project
    os.makedirs('outputs', exist_ok=True)
    try:
        shutil.copy2(_TMP_DB, DB_PATH)
    except Exception:
        pass  # outputs/ may be on a restricted filesystem; temp copy is still usable
    print(f"  [SQLite] Saved {len(df):,} rows → '{table_name}' ({_TMP_DB})")
    return _TMP_DB


def query(sql, params=None):
    """Execute a SQL query and return results as a DataFrame."""
    conn = get_connection()
    result = pd.read_sql_query(sql, conn, params=params or [])
    conn.close()
    return result
