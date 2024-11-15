import sympy as sp

mu, theta, kappa, rho, sigma, t, lambdaj, vj, muj = sp.symbols('mu theta kappa rho sigma t lambdaj vj muj')

Ert1 = (mu - theta/2)*t

Ert2 = (sigma * (- 4 * kappa * rho + sigma) * theta + sp.exp(kappa * t) * (- (sigma**2 * theta) - 4 * kappa**2 * rho * sigma * t * theta + kappa * sigma * (4 * rho + sigma * t) * theta +
        kappa**3 * t * (4 * theta + t * (-2 * mu + theta)**2 + lambdaj * vj * (4 + vj)) + 4 * kappa**3 * lambdaj * t * sp.log(1 + muj) * (-vj + sp.log(1 + muj)))) / (4 * sp.exp(kappa * t) * kappa**3)

Ert3 =  (- 3 * sigma * theta * (2 * sigma**3 + kappa * sigma**2 * (- 12 * rho + sigma * t) + 4 * kappa**3 * rho*(- 2 + 2 * mu * t + 2 * rho * sigma * t - t * theta) + 
                kappa**2 * sigma * (4 + 16 * rho**2 - 2 * mu * t - 6 * rho * sigma * t + t * theta)) + 
                sp.exp(kappa * t) * (6 * sigma**4 * theta - 3 * kappa * sigma**3 * (12 * rho + sigma * t) * theta + 12 * kappa**4 * rho * sigma * t * theta * (2 - 2 * mu * t + t * theta) + 
                3 * kappa**2 * sigma**2 * theta * (4 + 16 * rho**2 - 2 * mu * t + 6 * rho * sigma * t + t * theta) - 
                3 * kappa**3 * sigma * theta * (8 * rho**2 * sigma * t + sigma * t * (4 - 2 * mu * t + t * theta) + rho * (8 - 8 * mu * t + 4 * t * theta)) + 
                kappa**5 * t * (t * (2 * mu - theta) * (12 * theta + t * (- 2 * mu + theta)**2) + 12 * lambdaj * t * (2 * mu - theta) * vj + 3 * lambdaj * (- 4 + 2 * mu * t - t * theta) * vj**2 - 
                lambdaj * vj**3)) + 2 * sp.exp(kappa * t) * kappa**5 * lambdaj * t * sp.log(1 + muj) * (3 * vj * (4 - 4 * mu * t + 2 * t * theta + vj) - 6 * (-2 * mu * t + t * theta + vj) * sp.log(1 + muj) + 4 * sp.log(1 + muj)**2)) / (8 * sp.exp(kappa * t) * kappa**5)

Ert4 = (( - 24 * kappa**2 * (2 * kappa * rho - sigma) * sigma * t * (2 * (- 1 + sp.exp(kappa * t)) * sigma**2 - kappa * sigma * (8 * (- 1 + sp.exp(kappa * t)) * rho + sigma * t) 
           + 4 * kappa**2 * (- 1 + sp.exp(kappa * t) + rho * sigma * t)) * (2 * mu - theta) * theta)/sp.exp(kappa * t) + (3 * sigma**2 * theta * ((-4 * kappa*rho + sigma)**2 * 
           (sigma**2 + 2 * kappa * theta) + sp.exp(2 * kappa * t) * (- 32 * kappa**4 * (1 + 8 * rho**2) - 29 * sigma**4 + 2 * kappa * sigma**2 * (116 * rho * sigma + theta) - 
           16 * kappa**2 * sigma * (6 * sigma + 35 * rho**2 * sigma + rho * theta) + 32 * kappa**3 * rho * (12 * (1 + rho**2) * sigma + rho * theta)) - 
           4 * sp.exp(kappa * t) * (- 7 * sigma**4 + 16 * kappa**5 * rho**2 * t * (-2 + rho * sigma * t) + 4 * kappa**4 * (- 2 + rho * (12 * sigma*t +
           rho * (- 16 + sigma * t * (16 * rho - 5 * sigma * t)))) + kappa * sigma**2 * (56 * rho * sigma - 5 * sigma**2 * t + theta) - 
           kappa**2 * sigma * (sigma * (24 + 136 * rho**2 - 40 * rho * sigma * t + sigma**2 * t**2) + 8 * rho * theta) + 
           4 * kappa**3 * (sigma * (24 * (rho + rho**3) - 3 * (1 + 8 * rho**2) * sigma * t + 2 * rho * sigma**2 * t**2) + 4 * rho**2 * theta)))) / sp.exp(2 * kappa * t) + (12 * (- 1 + sp.exp(kappa * t)) * kappa**2 * (4 * kappa * rho - sigma) * sigma * t * theta * (- 4 * kappa * rho * sigma * theta + 
           sigma**2 * theta + kappa**2 * (4 * theta + t * (- 2 * mu + theta)**2 + lambdaj * vj * (4 + vj)) + 4 * kappa**2 * lambdaj * sp.log(1 + muj) * (- vj + sp.log(1 + muj)))) / sp.exp(kappa * t) + 
           2 * kappa * t * (- 120 * kappa * rho * sigma**5 * theta + 15 * sigma**6 * theta - 48 * kappa**3 * rho * sigma**3 * theta * (6 + 4 * rho**2 - 3 * mu * t + 2 * t * theta) + 
           3 * kappa**2 * sigma**4 * theta * (24 + 96 * rho**2 - 8 * mu * t + 5 * t * theta) + kappa**6 * (t * (16 * mu**4 * t**2 - 32 * mu**2 * t * (- 3 + mu * t) * theta +
           24 * (2 + mu * t * (- 4 + mu * t)) * theta**2 - 8 * t * (- 3 + mu * t) * theta**3 + t**2 * theta**4) + 24 * lambdaj * t * (4 * theta + t * (- 2 * mu + theta)**2) * vj + 
           6 * lambdaj * (8 + t * (8 * lambdaj + 4 * mu**2 * t - 4 * mu*(4 + t * theta) + theta * (12 + t * theta))) * vj**2 + 4 * lambdaj * (6 + t * (6 * lambdaj - 2 * mu + theta)) * vj**3 + 
           lambdaj * (1 + 3 * lambdaj * t) * vj**4) - 24 * kappa**5 * rho * sigma * t * theta * (4 * mu**2 * t - 4 * mu * (2 + t * theta) + theta * (8 + t * theta) + lambdaj * vj * (4 + vj)) + 
           6 * kappa**4 * sigma**2 * theta * (8 - 16 * mu * t + 8 * rho**2 * (4 - 4 * mu * t + 3 * t * theta) + t * (12 * theta + t * (- 2 * mu + theta)**2 + lambdaj * vj * (4 + vj))) + 
           8 * kappa**4 * lambdaj * sp.log(1 + muj) * (vj * (12 * kappa * rho * sigma * t * theta - 3 * sigma**2 * t * theta + 
           kappa**2 * (- 12 * mu**2 * t**2 - 3 * t * theta * (8 + t * theta) - 3 * (4 + 4 * lambdaj * t + t * theta) * vj - (1 + 3 * lambdaj * t) * vj**2 + 
           6 * mu * t*(4 + 2 * t * theta + vj))) + sp.log(1 + muj) * (3 * (- 4 * kappa * rho * sigma * t * theta + sigma**2 * t * theta + 
           kappa**2 * (t * (4 * theta + t * (-2 * mu + theta)**2) + 2 * (2 + t * (2 * lambdaj - 2 * mu + theta)) * vj + (1 + 3 * lambdaj * t) * vj**2)) + 
           2 * kappa**2 * sp.log(1 + muj) * (- 2 * (vj + t * (- 2 * mu + theta + 3 * lambdaj * vj)) + (1 + 3 * lambdaj * t) * sp.log(1 + muj)))))) / (32 * kappa**7)

Skewrt = (- 3 * (2 * kappa * rho - sigma) * sigma * (- 2 * sigma**2 + kappa * sigma * (8 * rho - sigma * t) + 4 * kappa**2 * (- 1 + rho * sigma * t)) * theta - 
             sp.exp(kappa * t) * (- 3 * (2 * kappa * rho - sigma) * sigma * (- 2 * sigma**2 + 4 * kappa**3 * t + kappa * sigma * (8 * rho + sigma * t) - 
             4 * kappa**2 * (1 + rho * sigma * t)) * theta + 12 * kappa**5 * lambdaj * t * vj**2 + kappa**5 * lambdaj * t * vj**3) + 
             2 * sp.exp(kappa * t) * kappa**5 * lambdaj * t * sp.log(1 + muj) * (3 * vj * (4 + vj) - 6 * vj * sp.log(1 + muj) + 4 * sp.log(1 + muj)**2)) / (sp.exp(kappa * t) * kappa**5 * (((- 1 + sp.exp(- kappa * t)) * sigma**2 * theta - 4 * kappa**2 * rho * sigma * t * theta + 
             kappa * sigma * ((4 - 4 / sp.exp(kappa * t)) * rho + sigma * t) * theta + kappa**3 * t * (4 * theta + lambdaj * vj * (4 + vj))) / kappa**3 - 
             4 * lambdaj * t * vj * sp.log(1 + muj) + 4 * lambdaj * t * sp.log(1 + muj)**2)**1.5)
             
Kurtrt = (3 * sigma**2 * (- 4 * kappa * rho + sigma)**2 * theta * (sigma**2 + 2 * kappa * theta) + 12 * sp.exp(kappa * t) * sigma * theta * (7 * sigma**5 - 
            kappa * sigma**3 * (56 * rho*sigma - 5 * sigma**2 * t + theta) + kappa**2 * sigma**2 * (- 40 * rho * sigma**2 * t + sigma**3 * t**2 + 
            8 * rho * theta + sigma * (24 + 136 * rho**2 + t * theta)) - 4 * kappa**3 * sigma * (24 * rho**3 * sigma - 3 * sigma**2 * t + 4 * rho**2 * (- 6 * sigma**2 * t + theta) + 
            2 * rho * sigma * (12 + sigma**2 * t**2 + t * theta)) - 4 * kappa**5 * rho * t * (- 8 * rho * sigma + 4 * rho**2 * sigma**2 * t + 4 * theta + lambdaj * vj * (4 + vj)) + 
            kappa**4 * sigma * (8 - 48 * rho * sigma * t - 64 * rho**3 * sigma * t + 4 * rho**2 * (16 + 5 * sigma**2 * t**2 + 4 * t * theta) + t * (4 * theta + lambdaj * vj * (4 + vj)))) + 
            sp.exp(2 * kappa * t) * (- 87 * sigma**6 * theta + 6 * kappa * sigma**4 * theta * (116 * rho * sigma + 5 * sigma**2 * t + theta) + 6 * kappa**3 * sigma**2 * theta * (192 * rho**3 * sigma + 
            16 * rho**2 * (6 * sigma**2 * t + theta) + 16 * rho * sigma * (12 + t * theta) + sigma**2 * t * (24 + t*theta)) - 12 * kappa**2 * sigma**3 * theta * (20 * rho * sigma**2 * t + 
            4 * rho * theta + sigma * (24 + 140 * rho**2 + t * theta)) - 48 * kappa**6 * rho * sigma * t**2 * theta * (4 * theta + lambdaj * vj * (4 + vj)) - 
            12 * kappa**4 * sigma**2 * theta * (8 + 32 * rho**3 * sigma * t + 16 * rho**2 * (4 + t * theta) + 4 * rho * sigma * t * (12 + t * theta) + 
            t * (4 * theta + lambdaj * vj * (4 + vj))) + 2 * kappa**7 * t * (lambdaj * vj**2 * (48 + 24 * vj + vj**2) + 3 * t * (4 * theta + lambdaj * vj * (4 + vj))**2) + 
            12 * kappa**5 * sigma * t * theta * (4 * rho * (4 * theta + lambdaj * vj * (4 + vj)) + sigma * (8 + 8 * rho**2 * (4 + t * theta) + t * (4 * theta + lambdaj * vj * (4 + vj))))) - 
            16 * sp.exp(kappa * t) * kappa**4 * lambdaj * t * vj * (- 3 * (- 1 + sp.exp(kappa * t)) * sigma**2 * theta - 12 * sp.exp(kappa * t) * kappa**2 * rho * sigma * t * theta + 
            3 * kappa * sigma * (4 * (- 1 + sp.exp(kappa * t)) * rho + sp.exp(kappa * t) * sigma * t) * theta + sp.exp(kappa * t) * kappa**3 * (vj * (12 + vj) +
            3 * t * (4 * theta + lambdaj * vj * (4 + vj)))) * sp.log(1 + muj) + 48 * sp.exp(kappa * t) * kappa**4 * lambdaj * t * ((1 - sp.exp(kappa * t)) * sigma**2 * theta - 
            4 * sp.exp(kappa * t) * kappa**2 * rho * sigma * t * theta + kappa * sigma * (4 * (- 1 + sp.exp(kappa * t)) * rho + sp.exp(kappa * t) * sigma * t) * theta + 
            sp.exp(kappa * t) * kappa**3 * (vj * (4 + vj) + t * (4 * theta + lambdaj * vj * (4 + 3 * vj)))) * sp.log(1 + muj)**2 - 
            64 * sp.exp(2 * kappa * t) * kappa**7 * lambdaj * t * (1 + 3 * lambdaj * t) * vj * sp.log(1 + muj)**3 + 32 * sp.exp(2 * kappa * t) * kappa**7 * lambdaj * t * (1 + 3 * lambdaj * t) * sp.log(1 + muj)**4) / (2 * sp.exp(2 * kappa * t) * kappa**7 * (((- 1 + sp.exp(- kappa * t)) * sigma**2 * theta - 4 * kappa**2 * rho * sigma * t * theta + 
            kappa * sigma * ((4 - 4 / sp.exp(kappa * t)) * rho + sigma * t) * theta + kappa**3 * t * (4 * theta + lambdaj * vj * (4 + vj))) / kappa**3 - 4 * lambdaj * t * vj * sp.log(1 + muj) + 
            4 * lambdaj * t * sp.log(1 + muj)**2)**2)

print(sp.simplify(Skewrt - ((Ert3 - 3 * Ert1 * Ert2 + 2 * Ert1**3) / ((Ert2 - Ert1**2)**(3/2)))))
print(sp.simplify(Kurtrt - ((Ert4 - 4 * Ert1 * Ert3 + 6 * Ert1**2 * Ert2 - 3 * Ert1**4) / ((Ert2 - Ert1**2)**2))))