
import numpy as np
import matplotlib.pyplot as plt

def fuzzy_membership_low(value, lower_bound, upper_bound):
    if value <= lower_bound:
        return 1
    elif lower_bound < value <= upper_bound:
        return (upper_bound - value) / (upper_bound - lower_bound)
    else:
        return 0

def fuzzy_membership_high(value, lower_bound, upper_bound):
    if value <= lower_bound:
        return 0
    elif lower_bound < value <= upper_bound:
        return (value - lower_bound) / (upper_bound - lower_bound)
    else:
        return 1

def fuzzy_membership_function(value, low_values, med_values, high_values):
    low = fuzzy_membership_low(value, *low_values)
    med = max(fuzzy_membership_low(value, *med_values), fuzzy_membership_high(value, *med_values))
    high = fuzzy_membership_high(value, *high_values)
    return low, med, high

def fuzzy_aggregation(memberships):
    return max(memberships)

def fuzzy_defuzzification(memberships, values):
    numerator = sum(m * v for m, v in zip(memberships, values))
    denominator = sum(m for m in memberships)
    return numerator / denominator

# Parâmetros para as funções de pertinência
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

criminalidade_low = (0, 10)
criminalidade_med = (5, 20)
criminalidade_high = (15, 30)

alerta_low = (0, 20)
alerta_med = (15, 40)
alerta_high = (35, 60)
alerta_muito_alto = (55, 80)
alerta_extremo = (75, 100)

# Exemplo numérico
densidade = 700
criminalidade = 18

# Fuzificação
membership_densidade = fuzzy_membership_function(densidade, densidade_low, densidade_med, densidade_high)
membership_criminalidade = fuzzy_membership_function(criminalidade, criminalidade_low, criminalidade_med, criminalidade_high)

# Operações AND e OR nas Regras
rule1 = fuzzy_aggregation([membership_densidade[0], membership_criminalidade[0]])
rule2 = fuzzy_aggregation([membership_densidade[2], membership_criminalidade[2]])
rule3 = fuzzy_aggregation([membership_densidade[1], membership_criminalidade[2]])

# Implicação (Método Produto)
implication1 = [min(rule1, alerta_low[0]), min(rule2, alerta_extremo[0]), min(rule3, alerta_muito_alto[0])]
implication2 = [min(rule1, alerta_low[1]), min(rule2, alerta_extremo[1]), min(rule3, alerta_muito_alto[1])]

# Agregação (Método Máximo)
aggregated_membership = fuzzy_aggregation(implication2)

# Defuzificação (Método Centroide)
defuzzified_value = fuzzy_defuzzification(implication2, [alerta_med[0], alerta_med[1], alerta_high[0]])

# Resultados
print(f"Membership Densidade: {membership_densidade}")
print(f"Membership Criminalidade: {membership_criminalidade}")
print(f"Regra 1 (Baixo): {rule1}")
print(f"Regra 2 (Extremo): {rule2}")
print(f"Regra 3 (Muito Alto): {rule3}")
print(f"Implicação (Método Produto): {implication1}, {implication2}")
print(f"Agregação (Método Máximo): {aggregated_membership}")
print(f"Defuzificação (Método Centroide): {defuzzified_value}")




