import sqlite3

# Create a connection to the database
conn = sqlite3.connect('simulation_database.db')
c = conn.cursor()

# Create the table 'simulations'
c.execute('''
    CREATE TABLE simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        max_number_of_same_prices INTEGER,
        NP_rc1 REAL,
        NP_rc2 REAL,
        NP_rc3 REAL,
        NP_rc4 REAL,
        NP_rm1 REAL,
        NP_rm2 REAL,
        NP_rm3 REAL,
        NP_rm4 REAL,
        NP_rskewness REAL,
        NP_rexcess_kurtosis REAL
    )
''')

conn.commit()