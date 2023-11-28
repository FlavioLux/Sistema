#plot das funções de entradas( Densidade populacional e Criminalidade )

import numpy as np
import matplotlib.pyplot as plt

def fuzzy_membership_triangular(value, lower_bound, peak, upper_bound):
    if lower_bound < value <= peak:
        return (value - lower_bound) / (peak - lower_bound)
    elif peak < value < upper_bound:
        return (upper_bound - value) / (upper_bound - peak)
    else:
        return 0

def fuzzy_membership_trapezoidal(value, left_bound, left_peak, right_peak, right_bound):
    if left_bound < value <= left_peak:
        return (value - left_bound) / (left_peak - left_bound)
    elif left_peak < value < right_peak:
        return 1
    elif right_peak < value < right_bound:
        return (right_bound - value) / (right_bound - right_peak)
    else:
        return 0

# Parâmetros para as funções de pertinência (exemplo com funções triangulares e trapezoidais)
densidade_low = (0, 200, 400)
densidade_med = (300, 600, 900)
densidade_high = (800, 1200, 1500)

criminalidade_low = (0, 5, 10)
criminalidade_med = (5, 15, 25)
criminalidade_high = (20, 30, 40)

# Exemplo numérico
densidade = 700
criminalidade = 18

# Fuzificação
x = np.linspace(0, 1600, 1000)

membership_densidade_low = [fuzzy_membership_triangular(val, *densidade_low) for val in x]
membership_densidade_med = [fuzzy_membership_triangular(val, *densidade_med) for val in x]
membership_densidade_high = [fuzzy_membership_triangular(val, *densidade_high) for val in x]

membership_criminalidade_low = [fuzzy_membership_triangular(val, *criminalidade_low) for val in x]
membership_criminalidade_med = [fuzzy_membership_triangular(val, *criminalidade_med) for val in x]
membership_criminalidade_high = [fuzzy_membership_triangular(val, *criminalidade_high) for val in x]

# Visualização das funções de pertinência
plt.figure(figsize=(12, 8))

# Funções de pertinência para Densidade
plt.subplot(2, 1, 1)
plt.plot(x, membership_densidade_low, label='Densidade Low')
plt.plot(x, membership_densidade_med, label='Densidade Medium')
plt.plot(x, membership_densidade_high, label='Densidade High')
plt.title('Membership Functions - Densidade')
plt.legend()

# Funções de pertinência para Criminalidade
plt.subplot(2, 1, 2)
plt.plot(x, membership_criminalidade_low, label='Criminalidade Low')
plt.plot(x, membership_criminalidade_med, label='Criminalidade Medium')
plt.plot(x, membership_criminalidade_high, label='Criminalidade High')
plt.title('Membership Functions - Criminalidade')
plt.legend()

plt.tight_layout()
plt.show()
