import sqlite3

def add_column(table, column, column_type):
    conn = sqlite3.connect('simulations.db')
    cursor = conn.cursor()
    
    # check if column exists
    cursor.execute(f'PRAGMA table_info({table})')
    columns = cursor.fetchall()
    columns = [column[1] for column in columns]
    if column in columns:
        print(f'Column {column} already exists in table {table}')
    else:
        cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}')
        conn.commit()
    conn.close()
    
def update_value(table, column, value, mu, kappa, theta, sigma, rho, v0):
    conn = sqlite3.connect('simulations.db')
    cursor = conn.cursor()
    
    where_stmt = f'mu={mu} AND kappa={kappa} AND theta={theta} AND sigma={sigma} AND rho={rho} AND v0={v0}'
    
    # check if only value would be updated
    cursor.execute(f'SELECT count(*) AS num FROM {table} WHERE {where_stmt}')
    num = cursor.fetchone()[0]
    if num == 0:
        print(f'No entry found for {table} with {where_stmt}')
        return
    elif num > 1:
        print(f'Multiple entries found for {table} with {where_stmt}')
        return
    elif num == 1:
        pass
    
    # update value
    cursor.execute(f'SELECT {column} FROM {table} WHERE {where_stmt}')
    old_value = cursor.fetchone()
    if old_value == value:
        print(f'Value {value} already exists in column {column} for {table} with {where_stmt}')
    else:
        cursor.execute(f'UPDATE {table} SET {column}={value} WHERE {where_stmt}')
        conn.commit()