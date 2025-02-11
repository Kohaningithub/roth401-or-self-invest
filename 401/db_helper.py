import sqlite3
from pathlib import Path
import streamlit as st

def init_db():
    """Initialize SQLite database"""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Create and connect to database
    db_path = data_dir / 'user_calculations.db'
    print(f"Creating database at: {db_path}")  # Debug print
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS calculations
                 (user_id TEXT,
                  timestamp TEXT,
                  current_age INTEGER,
                  annual_income REAL,
                  state TEXT,
                  retirement_age INTEGER,
                  roth_contribution REAL,
                  employer_match REAL,
                  match_limit REAL,
                  inflation_rate REAL,
                  salary_growth REAL,
                  401k_return REAL,
                  active_return REAL,
                  passive_return REAL,
                  roth_value REAL,
                  self_managed_value REAL,
                  difference REAL)''')
    conn.commit()
    conn.close()

def save_calculation(data):
    """Save calculation to database"""
    db_path = Path(__file__).parent / 'data' / 'user_calculations.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    sql = f'INSERT INTO calculations ({columns}) VALUES ({placeholders})'
    
    c.execute(sql, list(data.values()))
    conn.commit()
    conn.close()

def get_user_calculations(user_id):
    """Get all calculations for a user"""
    db_path = Path(__file__).parent / 'data' / 'user_calculations.db'
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query('SELECT * FROM calculations WHERE user_id = ? ORDER BY timestamp DESC', 
                          conn, 
                          params=(user_id,))
    conn.close()
    return df

def clear_user_history(user_id):
    """Delete all calculations for a user"""
    db_path = Path(__file__).parent / 'data' / 'user_calculations.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('DELETE FROM calculations WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()