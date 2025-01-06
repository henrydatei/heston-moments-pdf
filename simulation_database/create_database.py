import sqlite3

# Create a connection to the database
conn = sqlite3.connect('simulation_database.db')
c = conn.cursor()

# Create the table 'simulations'
c.execute('''
    CREATE TABLE simulations (
        id INTEGER PRIMARY KEY,
        mu REAL,
        kappa REAL,
        theta REAL,
        sigma REAL,
        rho REAL,
        v0 REAL,
        time_points INTEGER,
        burnin INTEGER,
        T INTEGER,
        S0 REAL,
        paths INTEGER,
        start_date TEXT,
        end_date TEXT,
        interval TEXT,
        filename TEXT
    )
''')

conn.commit()