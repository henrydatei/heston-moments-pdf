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

def update_multiple_values(table, values, mu, kappa, theta, sigma, rho, v0):
    """
    Aktualisiert mehrere Spalten in einer Tabelle, sofern genau ein Datensatz 
    mit den angegebenen Parametern existiert.

    Parameter:
      table (str): Name der Tabelle.
      values (dict): Dictionary mit Spaltennamen als Schlüssel und den neuen Werten als Wert.
      mu, kappa, theta, sigma, rho, v0: Parameter, die in der WHERE-Klausel verwendet werden.
    """
    conn = sqlite3.connect('simulations.db')
    cursor = conn.cursor()
    
    # WHERE-Klausel und Parameterbindung
    where_clause = "mu=? AND kappa=? AND theta=? AND sigma=? AND rho=? AND v0=?"
    where_params = (mu, kappa, theta, sigma, rho, v0)
    
    # Überprüfen, ob genau ein Eintrag existiert
    cursor.execute(f"SELECT count(*) FROM {table} WHERE {where_clause}", where_params)
    num = cursor.fetchone()[0]
    if num == 0:
        print(f'Kein Eintrag gefunden in {table} mit {where_clause} und Werten {where_params}')
        return
    elif num > 1:
        print(f'Mehrere Einträge gefunden in {table} mit {where_clause} und Werten {where_params}')
        return
    
    # Abfrage der aktuellen Werte für die zu aktualisierenden Spalten
    columns = ", ".join(values.keys())
    cursor.execute(f"SELECT {columns} FROM {table} WHERE {where_clause}", where_params)
    current_values = cursor.fetchone()
    
    # Prüfen, welche Spalten aktualisiert werden müssen
    changes = {}
    for i, col in enumerate(values.keys()):
        if current_values[i] != values[col]:
            changes[col] = values[col]
    
    if changes:
        # Aufbau der SET-Klausel mit Parameterbindung
        set_clause = ", ".join([f"{col}=?" for col in changes.keys()])
        update_params = list(changes.values()) + list(where_params)
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        cursor.execute(query, update_params)
        conn.commit()
        # print(f"Spalten aktualisiert: {', '.join(changes.keys())}")
    
    conn.close()
