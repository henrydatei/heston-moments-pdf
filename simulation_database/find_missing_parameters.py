import itertools
import numpy as np
import sqlite3

# Definition der Parametergrenzen und -schritte
v0_min, v0_max, v0_step = 0.01, 0.5, 0.05
kappa_min, kappa_max, kappa_step = 0.01, 1, 0.1
theta_min, theta_max, theta_step = 0.01, 1, 0.05
sigma_min, sigma_max, sigma_step = 0.01, 1, 0.05
rho_min, rho_max, rho_step = -0.9, 0.9, 0.1

# Erzeugen der Arrays
v0s = np.arange(v0_min, v0_max, v0_step)
kappas = np.arange(kappa_min, kappa_max, kappa_step)
thetas = np.arange(theta_min, theta_max, theta_step)
sigmas = np.arange(sigma_min, sigma_max, sigma_step)
mus = [0]
rhos = np.arange(rho_min, rho_max, rho_step)

# Erzeugen des vollständigen Parameterraums
# (Bei Bedarf die Werte runden, um Rundungsprobleme zu vermeiden)
def runden(x, ndigits=6):
    return round(x, ndigits)

all_combinations = set(
    (runden(v0), runden(kappa), runden(theta), runden(sigma), runden(mu), runden(rho))
    for v0, kappa, theta, sigma, mu, rho in itertools.product(v0s, kappas, thetas, sigmas, mus, rhos)
)

print("Erwartete Anzahl Kombinationen:", len(all_combinations))

# Verbindung zur SQLite-Datenbank herstellen
conn = sqlite3.connect("simulations.db")
cursor = conn.cursor()

# Auslesen der in der Datenbank gespeicherten Parameterkombinationen
cursor.execute("SELECT v0, kappa, theta, sigma, mu, rho FROM simulations where mu = 0")
db_combinations = set(
    (runden(row[0]), runden(row[1]), runden(row[2]), runden(row[3]), runden(row[4]), runden(row[5]))
    for row in cursor.fetchall()
)

print("In der DB gefundene Kombinationen:", len(db_combinations))

# Fehlende Kombinationen ermitteln
missing = all_combinations - db_combinations

print("Anzahl fehlender Kombinationen:", len(missing))
print("Fehlende Kombinationen:")
for comb in sorted(missing):
    print(comb)

conn.close()
