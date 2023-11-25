#!/usr/bin/env python
# coding: utf-8

# In[2]:


#%matplotlib inline
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




# In[9]:


get_ipython().system('pip install --upgrade matplotlib')


# In[ ]:





# In[ ]:





# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
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




# In[12]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

criminalidade_low = (0, 10)
criminalidade_med = (5, 20)
criminalidade_high = (15, 30)

# Criando um array de valores x no intervalo para a Densidade Populacional
x_densidade = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Densidade Populacional
y_densidade_low = [fuzzy_membership_low(x, *densidade_low) for x in x_densidade]
y_densidade_med = [fuzzy_membership_function(x, densidade_low, densidade_med, densidade_high)[1] for x in x_densidade]
y_densidade_high = [fuzzy_membership_high(x, *densidade_high) for x in x_densidade]

# Criando um array de valores x no intervalo para a Taxa de Criminalidade
x_criminalidade = np.linspace(0, 35, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Taxa de Criminalidade
y_criminalidade_low = [fuzzy_membership_low(x, *criminalidade_low) for x in x_criminalidade]
y_criminalidade_med = [fuzzy_membership_function(x, criminalidade_low, criminalidade_med, criminalidade_high)[1] for x in x_criminalidade]
y_criminalidade_high = [fuzzy_membership_high(x, *criminalidade_high) for x in x_criminalidade]

# Plotando os gráficos
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(x_densidade, y_densidade_low, label='Baixa')
plt.title('Densidade Populacional - Baixa')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_densidade, y_densidade_med, label='Média')
plt.title('Densidade Populacional - Média')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_densidade, y_densidade_high, label='Alta')
plt.title('Densidade Populacional - Alta')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_criminalidade, y_criminalidade_low, label='Baixa')
plt.plot(x_criminalidade, y_criminalidade_med, label='Média')
plt.plot(x_criminalidade, y_criminalidade_high, label='Alta')
plt.title('Taxa de Criminalidade')
plt.xlabel('Taxa de Criminalidade (crimes/mês)')
plt.ylabel('Pertinência')
plt.legend()

plt.tight_layout()
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da Densidade Populacional
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

# Criando um array de valores x no intervalo para a Densidade Populacional
x_densidade = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Densidade Populacional
y_densidade_low = [fuzzy_membership_low(x, *densidade_low) for x in x_densidade]
y_densidade_med = [fuzzy_membership_function(x, densidade_low, densidade_med, densidade_high)[1] for x in x_densidade]
y_densidade_high = [fuzzy_membership_high(x, *densidade_high) for x in x_densidade]

# Plotando os gráficos
plt.figure(figsize=(8, 5))

plt.plot(x_densidade, y_densidade_low, label='Baixa')
plt.plot(x_densidade, y_densidade_med, label='Média')
plt.plot(x_densidade, y_densidade_high, label='Alta')

plt.title('Funções de Pertinência - Densidade Populacional')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da Densidade Populacional
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

# Criando um array de valores x no intervalo para a Densidade Populacional
x_densidade = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Densidade Populacional
y_densidade_low = [fuzzy_membership_low(x, *densidade_low) for x in x_densidade]
y_densidade_med = [fuzzy_membership_function(x, densidade_low, densidade_med, densidade_high)[0] for x in x_densidade]
y_densidade_high = [fuzzy_membership_high(x, *densidade_high) for x in x_densidade]

# Plotando os gráficos
plt.figure(figsize=(8, 5))

plt.plot(x_densidade, y_densidade_low, label='Baixa')
plt.plot(x_densidade, y_densidade_med, label='Média')
plt.plot(x_densidade, y_densidade_high, label='Alta')

plt.title('Funções de Pertinência - Densidade Populacional')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da Densidade Populacional
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

# Criando um array de valores x no intervalo para a Densidade Populacional
x_densidade = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Densidade Populacional
y_densidade_low = [fuzzy_membership_low(x, *densidade_low) for x in x_densidade]
y_densidade_med = [fuzzy_membership_function(x, densidade_low, densidade_med, densidade_high)[1] for x in x_densidade]
y_densidade_high = [fuzzy_membership_high(x, *densidade_high) for x in x_densidade]

# Plotando os gráficos
plt.figure(figsize=(8, 5))

plt.plot(x_densidade, y_densidade_low, label='Baixa')
plt.plot(x_densidade, y_densidade_med, label='Média')
plt.plot(x_densidade, y_densidade_high, label='Alta')

plt.title('Funções de Pertinência - Densidade Populacional')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da Densidade Populacional
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

# Criando um array de valores x no intervalo para a Densidade Populacional
x_densidade = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência da Densidade Populacional
y_densidade_low = [fuzzy_membership_low(x, *densidade_low) for x in x_densidade]
y_densidade_med = [fuzzy_membership_function(x, densidade_low, densidade_med, densidade_high)[0] for x in x_densidade]
y_densidade_high = [fuzzy_membership_high(x, *densidade_high) for x in x_densidade]

# Plotando os gráficos
plt.figure(figsize=(8, 5))

plt.plot(x_densidade, y_densidade_low, label='Baixa')
plt.plot(x_densidade, y_densidade_med, label='Média')
plt.plot(x_densidade, y_densidade_high, label='Alta')

plt.title('Funções de Pertinência - Densidade Populacional')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da Densidade Populacional
densidade_low = (0, 500)
densidade_med = (400, 1000)
densidade_high = (800, 1500)

# Parâmetros para as funções de pertinência da Taxa de Criminalidade
criminalidade_low = (0, 10)
criminalidade_med = (5, 20)
criminalidade_high = (15, 30)

# Parâmetros para as funções de pertinência da Latência Criminal (faixas fictícias, ajuste conforme necessário)
latencia_low = (0, 200)
latencia_med = (150, 400)
latencia_high = (350, 600)

# Criando um array de valores x no intervalo para as variáveis
x = np.linspace(0, 2000, 1000)

# Calculando os valores correspondentes de y para cada função de pertinência
y_densidade_low = [fuzzy_membership_low(val, *densidade_low) for val in x]
y_densidade_med = [fuzzy_membership_function(val, densidade_low, densidade_med, densidade_high)[0] for val in x]
y_densidade_high = [fuzzy_membership_high(val, *densidade_high) for val in x]

y_criminalidade_low = [fuzzy_membership_low(val, *criminalidade_low) for val in x]
y_criminalidade_med = [fuzzy_membership_function(val, criminalidade_low, criminalidade_med, criminalidade_high)[0] for val in x]
y_criminalidade_high = [fuzzy_membership_high(val, *criminalidade_high) for val in x]

y_latencia_low = [fuzzy_membership_low(val, *latencia_low) for val in x]
y_latencia_med = [fuzzy_membership_function(val, latencia_low, latencia_med, latencia_high)[0] for val in x]
y_latencia_high = [fuzzy_membership_high(val, *latencia_high) for val in x]

# Plotando os gráficos
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x, y_densidade_low, label='Baixa')
plt.plot(x, y_densidade_med, label='Média')
plt.plot(x, y_densidade_high, label='Alta')
plt.title('Funções de Pertinência - Densidade Populacional')
plt.xlabel('Densidade Populacional (hab/km²)')
plt.ylabel('Pertinência')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, y_criminalidade_low, label='Baixa')
plt.plot(x, y_criminalidade_med, label='Média')
plt.plot(x, y_criminalidade_high, label='Alta')
plt.title('Funções de Pertinência - Taxa de Criminalidade')
plt.xlabel('Taxa de Criminalidade (crimes/mês)')
plt.ylabel('Pertinência')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, y_latencia_low, label='Baixa')
plt.plot(x, y_latencia_med, label='Média')
plt.plot(x, y_latencia_high, label='Alta')
plt.title('Funções de Pertinência - Latência Criminal')
plt.xlabel('Latência Criminal')
plt.ylabel('Pertinência')
plt.legend()

plt.tight_layout()
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Valores para a variável de entrada Criminalidade
criminalidade_values = np.linspace(0, 35, 1000)

# Calcula os graus de pertinência para cada conjunto fuzzy da variável de entrada Criminalidade
criminalidade_low_values = [fuzzy_membership_low(val, *criminalidade_low) for val in criminalidade_values]
criminalidade_med_values = [fuzzy_membership_function(val, criminalidade_low, criminalidade_med, criminalidade_high)[0] for val in criminalidade_values]
criminalidade_high_values = [fuzzy_membership_high(val, *criminalidade_high) for val in criminalidade_values]

# Plotando os gráficos
plt.figure(figsize=(10, 6))

plt.plot(criminalidade_values, criminalidade_low_values, label='Baixa')
plt.plot(criminalidade_values, criminalidade_med_values, label='Média')
plt.plot(criminalidade_values, criminalidade_high_values, label='Alta')

plt.title('Funções de Pertinência - Criminalidade')
plt.xlabel('Criminalidade (crimes/mês)')
plt.ylabel('Pertinência')
plt.legend()

plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros para as funções de pertinência da variável de saída Alerta de Segurança
alerta_values = np.linspace(0, 100, 1000)

# Calcula os graus de pertinência para cada conjunto fuzzy da variável de saída Alerta de Segurança
alerta_low_values = [fuzzy_membership_low(val, *alerta_low) for val in alerta_values]
alerta_med_values = [fuzzy_membership_function(val, alerta_low, alerta_med, alerta_high)[0] for val in alerta_values]
alerta_high_values = [fuzzy_membership_high(val, *alerta_high) for val in alerta_values]

# Plotando os gráficos
plt.figure(figsize=(10, 6))

plt.plot(alerta_values, alerta_low_values, label='Baixo')
plt.plot(alerta_values, alerta_med_values, label='Médio')
plt.plot(alerta_values, alerta_high_values, label='Alto')

plt.title('Funções de Pertinência - Alerta de Segurança')
plt.xlabel('Pertinência')
plt.ylabel('Alerta de Segurança')
plt.legend()

plt.show()


# In[21]:


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

def fuzzy_membership_function(value, low_values, med_values, high_values, very_high_values, extreme_values):
    low = fuzzy_membership_low(value, *low_values)
    med = max(fuzzy_membership_low(value, *med_values), fuzzy_membership_high(value, *med_values))
    high = max(fuzzy_membership_high(value, *high_values), fuzzy_membership_low(value, *very_high_values))
    very_high = max(fuzzy_membership_low(value, *very_high_values), fuzzy_membership_high(value, *very_high_values))
    extreme = fuzzy_membership_high(value, *extreme_values)
    return low, med, high, very_high, extreme

def fuzzy_aggregation(memberships):
    return max(memberships)

def fuzzy_defuzzification(memberships, values):
    numerator = sum(m * v for m, v in zip(memberships, values))
    denominator = sum(m for m in memberships)
    return numerator / denominator

# Parâmetros para as funções de pertinência
alerta_low = (0, 20)
alerta_med = (15, 40)
alerta_high = (35, 60)
alerta_very_high = (55, 80)
alerta_extreme = (75, 100)

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
implication1 = [min(rule1, alerta_low[0]), min(rule2, alerta_extreme[0]), min(rule3, alerta_very_high[0])]
implication2 = [min(rule1, alerta_low[1]), min(rule2, alerta_extreme[1]), min(rule3, alerta_very_high[1])]

# Agregação (Método Máximo)
aggregated_membership = fuzzy_aggregation(implication2)

# Defuzificação (Método Centroide)
defuzzified_value = fuzzy_defuzzification(implication2, [alerta_med[0], alerta_med[1], alerta_high[0], alerta_high[1], alerta_very_high[0]])

# Resultados
print(f"Membership Densidade: {membership_densidade}")
print(f"Membership Criminalidade: {membership_criminalidade}")
print(f"Regra 1 (Baixo): {rule1}")
print(f"Regra 2 (Extremo): {rule2}")
print(f"Regra 3 (Muito Alto): {rule3}")
print(f"Implicação (Método Produto): {implication1}, {implication2}")
print(f"Agregação (Método Máximo): {aggregated_membership}")
print(f"Defuzificação (Método Centroide): {defuzzified_value}")




# In[23]:


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

def fuzzy_membership_function(value, low_values, med_values, high_values, very_high_values, extreme_values):
    low = fuzzy_membership_low(value, *low_values)
    med = max(fuzzy_membership_low(value, *med_values), fuzzy_membership_high(value, *med_values))
    high = max(fuzzy_membership_high(value, *high_values), fuzzy_membership_low(value, *very_high_values))
    very_high = max(fuzzy_membership_low(value, *very_high_values), fuzzy_membership_high(value, *very_high_values))
    extreme = fuzzy_membership_high(value, *extreme_values)
    return low, med, high, very_high, extreme

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
alerta_very_high = (55, 80)
alerta_extreme = (75, 100)

# Exemplo numérico
densidade = 700
criminalidade = 18

# Fuzificação
membership_densidade = fuzzy_membership_function(densidade, densidade_low, densidade_med, densidade_high, alerta_very_high, alerta_extreme)
membership_criminalidade = fuzzy_membership_function(criminalidade, criminalidade_low, criminalidade_med, criminalidade_high, alerta_very_high, alerta_extreme)

# Operações AND e OR nas Regras
rule1 = fuzzy_aggregation([min(membership_densidade[0], membership_criminalidade[0])])
rule2 = fuzzy_aggregation([min(membership_densidade[2], membership_criminalidade[2])])
rule3 = fuzzy_aggregation([max(membership_densidade[1], membership_criminalidade[2])])

# Implicação (Método Produto)
implication1 = [min(rule1, alerta_low[0]), min(rule2, alerta_extreme[0]), min(rule3, alerta_very_high[0])]
implication2 = [min(rule1, alerta_low[1]), min(rule2, alerta_extreme[1]), min(rule3, alerta_very_high[1])]

# Agregação (Método Máximo)
aggregated_membership = fuzzy_aggregation(implication2)

# Defuzificação (Método Centroide)
defuzzified_value = fuzzy_defuzzification(implication2, [alerta_med[0], alerta_med[1], alerta_high[0], alerta_high[1], alerta_very_high[0]])

# Resultados
print(f"Membership Densidade: {membership_densidade}")
print(f"Membership Criminalidade: {membership_criminalidade}")
print(f"Regra 1 (Baixo): {rule1}")
print(f"Regra 2 (Extremo): {rule2}")
print(f"Regra 3 (Muito Alto): {rule3}")
print(f"Implicação (Método Produto): {implication1}, {implication2}")
print(f"Agregação (Método Máximo): {aggregated_membership}")
print(f"Defuzificação (Método Centroide): {defuzzified_value}")


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# Função de pertinência triangular
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Variáveis de entrada e saída
x_values = np.arange(0, 11, 0.1)
y_values_error = {'Negativo Grande': triangular_mf(x_values, -5, -2.5, 0),
                  'Negativo Médio': triangular_mf(x_values, -2.5, -1, 0),
                  'Zero': triangular_mf(x_values, -1, 1, 1),
                  'Positivo Médio': triangular_mf(x_values, 0, 1, 2.5),
                  'Positivo Grande': triangular_mf(x_values, 0, 2.5, 5)}

y_values_derivative = {'Negativo Grande': triangular_mf(x_values, -2.5, -1, 0),
                       'Negativo Médio': triangular_mf(x_values, -1, -0.5, 0),
                       'Zero': triangular_mf(x_values, -0.5, 0.5, 0.5),
                       'Positivo Médio': triangular_mf(x_values, 0, 0.5, 1),
                       'Positivo Grande': triangular_mf(x_values, 0, 1, 2.5)}

y_values_valve = {'Pequena': triangular_mf(x_values, 0, 20, 40),
                  'Média': triangular_mf(x_values, 20, 50, 80),
                  'Grande': triangular_mf(x_values, 60, 80, 100)}

# Plotagem
plt.figure(figsize=(12, 8))

# Funções de pertinência para o sinal de erro
plt.subplot(3, 1, 1)
for label, values in y_values_error.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Sinal de Erro')
plt.legend()

# Funções de pertinência para a derivada do erro
plt.subplot(3, 1, 2)
for label, values in y_values_derivative.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Derivada do Erro')
plt.legend()

# Funções de pertinência para a abertura da válvula
plt.subplot(3, 1, 3)
for label, values in y_values_valve.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Abertura da Válvula')
plt.legend()

plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt

# Função de pertinência triangular
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Variáveis de entrada e saída
x_values = np.arange(0, 11, 0.1)

# Funções de pertinência para o sinal de erro
y_values_error = {'Negativo Grande': triangular_mf(x_values, -5, -2.5, 0),
                  'Negativo Médio': triangular_mf(x_values, -2.5, -1, 0),
                  'Zero': triangular_mf(x_values, -1, 1, 1),
                  'Positivo Médio': triangular_mf(x_values, 0, 1, 2.5),
                  'Positivo Grande': triangular_mf(x_values, 0, 2.5, 5)}

# Funções de pertinência para a derivada do erro
y_values_derivative = {'Negativo Grande': triangular_mf(x_values, -2.5, -1, 0),
                       'Negativo Médio': triangular_mf(x_values, -1, -0.5, 0),
                       'Zero': triangular_mf(x_values, -0.5, 0.5, 0.5),
                       'Positivo Médio': triangular_mf(x_values, 0, 0.5, 1),
                       'Positivo Grande': triangular_mf(x_values, 0, 1, 2.5)}

# Funções de pertinência para a abertura da válvula
y_values_valve = {'Pequena': triangular_mf(x_values, 0, 20, 40),
                  'Média': triangular_mf(x_values, 20, 50, 80),
                  'Grande': triangular_mf(x_values, 60, 80, 100)}

# Plotagem
plt.figure(figsize=(12, 8))

# Funções de pertinência para o sinal de erro
plt.subplot(3, 1, 1)
for label, values in y_values_error.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Sinal de Erro')
plt.legend()

# Funções de pertinência para a derivada do erro
plt.subplot(3, 1, 2)
for label, values in y_values_derivative.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Derivada do Erro')
plt.legend()

# Funções de pertinência para a abertura da válvula
plt.subplot(3, 1, 3)
for label, values in y_values_valve.items():
    plt.plot(x_values, values, label=label)
plt.title('Funções de Pertinência - Abertura da Válvula')
plt.legend()

plt.tight_layout()
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# Funções de pertinência para a variável de saída
output_values = np.arange(0, 101, 1)
membership_output = [fuzzy_membership_high(value, *alerta_low) for value in output_values]
membership_output = np.maximum(membership_output, [fuzzy_membership_low(value, *alerta_med) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_low(value, *alerta_high) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_high(value, *alerta_very_high) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_high(value, *alerta_extreme) for value in output_values])

# Valor defuzzificado
defuzzified_output = fuzzy_defuzzification([aggregated_membership], [defuzzified_value])

# Plotagem
plt.figure(figsize=(10, 6))

# Funções de pertinência
plt.plot(output_values, membership_output, label='Funções de Pertinência')
plt.vlines(defuzzified_output, 0, 1, colors='r', linestyles='dashed', label='Valor Defuzzificado')

# Configurações do gráfico
plt.title('Saída - Alerta de Segurança')
plt.xlabel('Valor da Saída')
plt.ylabel('Pertinência')
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


import matplotlib.pyplot as plt
import numpy as np

# Funções de pertinência para a variável de saída
output_values = np.arange(0, 101, 1)
membership_output = [fuzzy_membership_high(value, *alerta_low) for value in output_values]
membership_output = np.maximum(membership_output, [fuzzy_membership_low(value, *alerta_med) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_low(value, *alerta_high) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_high(value, *alerta_very_high) for value in output_values])
membership_output = np.maximum(membership_output, [fuzzy_membership_high(value, *alerta_extreme) for value in output_values])

# Valor defuzzificado
defuzzified_output = fuzzy_defuzzification([aggregated_membership], [defuzzified_value])

# Plotagem
plt.figure(figsize=(10, 6))

# Área sob a curva até o ponto de defuzzificação
plt.fill_between(output_values, 0, membership_output, where=(output_values <= defuzzified_output), interpolate=True, alpha=0.3, label='Área Sob a Curva')

# Linha da curva
plt.plot(output_values, membership_output, label='Funções de Pertinência')

# Linha vertical indicando o valor defuzzificado
plt.vlines(defuzzified_output, 0, 1, colors='r', linestyles='dashed', label='Valor Defuzzificado')

# Configurações do gráfico
plt.title('Saída - Alerta de Segurança')
plt.xlabel('Valor da Saída')
plt.ylabel('Pertinência')
plt.legend()
plt.grid(True)
plt.show()


# In[28]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variáveis de entrada
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')

# Variável de saída
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

# Funções de pertinência para as variáveis de entrada e saída
servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['bom'] = fuzz.trimf(servico.universe, [0, 5, 10])

comida['pessima'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['boa'] = fuzz.trimf(comida.universe, [0, 5, 10])

gorjeta['pequena'] = fuzz.trimf(gorjeta.universe, [0, 0, 12])
gorjeta['mediana'] = fuzz.trimf(gorjeta.universe, [0, 12, 25])
gorjeta['generosa'] = fuzz.trimf(gorjeta.universe, [12, 25, 25])

# Regras
regra1 = ctrl.Rule(servico['ruim'] & comida['pessima'], gorjeta['pequena'])
regra2 = ctrl.Rule(servico['bom'] | comida['boa'], gorjeta['generosa'])
regra3 = ctrl.Rule(servico['bom'] & comida['pessima'], gorjeta['mediana'])

# Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Entradas (valores entre 0 e 10)
sistema.input['servico'] = 7
sistema.input['comida'] = 8

# Computar o resultado
sistema.compute()

# Saída
print("Gorjeta:", sistema.output['gorjeta'])
gorjeta.view(sim=sistema)


# In[29]:


pip install scikit-fuzzy


# In[30]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variáveis de entrada
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')

# Variável de saída
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

# Funções de pertinência para as variáveis de entrada e saída
servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['bom'] = fuzz.trimf(servico.universe, [0, 5, 10])

comida['pessima'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['boa'] = fuzz.trimf(comida.universe, [0, 5, 10])

gorjeta['pequena'] = fuzz.trimf(gorjeta.universe, [0, 0, 12])
gorjeta['mediana'] = fuzz.trimf(gorjeta.universe, [0, 12, 25])
gorjeta['generosa'] = fuzz.trimf(gorjeta.universe, [12, 25, 25])

# Regras
regra1 = ctrl.Rule(servico['ruim'] & comida['pessima'], gorjeta['pequena'])
regra2 = ctrl.Rule(servico['bom'] | comida['boa'], gorjeta['generosa'])
regra3 = ctrl.Rule(servico['bom'] & comida['pessima'], gorjeta['mediana'])

# Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Entradas (valores entre 0 e 10)
sistema.input['servico'] = 7
sistema.input['comida'] = 8

# Computar o resultado
sistema.compute()

# Saída
print("Gorjeta:", sistema.output['gorjeta'])
gorjeta.view(sim=sistema)


# In[31]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Variáveis de entrada
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')

# Variável de saída
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

# Funções de pertinência para as variáveis de entrada e saída
servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['medio'] = fuzz.trimf(servico.universe, [0, 5, 10])
servico['bom'] = fuzz.trimf(servico.universe, [5, 10, 10])

comida['pessima'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['media'] = fuzz.trimf(comida.universe, [0, 5, 10])
comida['boa'] = fuzz.trimf(comida.universe, [5, 10, 10])

gorjeta['pequena'] = fuzz.trimf(gorjeta.universe, [0, 0, 12])
gorjeta['mediana'] = fuzz.trimf(gorjeta.universe, [0, 12, 25])
gorjeta['generosa'] = fuzz.trimf(gorjeta.universe, [12, 25, 25])

# Regras
regra1 = ctrl.Rule(servico['ruim'] & comida['pessima'], gorjeta['pequena'])
regra2 = ctrl.Rule(servico['bom'] | comida['boa'], gorjeta['generosa'])
regra3 = ctrl.Rule(servico['medio'] & comida['media'], gorjeta['mediana'])

# Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Entradas (valores entre 0 e 10)
sistema.input['servico'] = 7
sistema.input['comida'] = 8

# Computar o resultado
sistema.compute()

# Saída
print("Gorjeta:", sistema.output['gorjeta'])

# Visualização dos resultados
gorjeta.view(sim=sistema)
plt.show()


# In[ ]:




