"""
Este script realiza as seguintes tarefas:
1. Busca um conjunto de dados do UC Irvine Machine Learning Repository.
2. Renomeia as colunas do conjunto de dados.
3. Verifica valores ausentes e os preenche com a moda de cada coluna.
4. Gera combinações de atributos e calcula probabilidades básicas para essas combinações.
5. Salva as probabilidades calculadas em um arquivo.
6. Transforma o conjunto de dados em um formato binário e aplica o algoritmo Apriori para encontrar conjuntos frequentes de itens.
7. Gera regras de associação a partir dos conjuntos frequentes de itens e as salva em um arquivo.
8. Calcula intervalos de confiança de Wald para as probabilidades e os salva em um arquivo.
Funções:
- computeBasicProb(combination, operator): Calcula probabilidades básicas para combinações de atributos usando operadores "and" ou "or".
- compute_wald_confidence_interval(prob, n, confidence=0.95): Calcula intervalos de confiança de Wald para probabilidades dadas.
Dependências:
- mlxtend
- pandas
- string
- ipdb
- numpy
- scipy
- sys
- itertools
- ucimlrepo
Uso:
- Certifique-se de que todas as dependências estão instaladas usando o seguinte comando:
    pip install mlxtend pandas ipdb numpy scipy ucimlrepo
- Execute o script para buscar o conjunto de dados, processá-lo e gerar probabilidades, regras de associação e intervalos de confiança.
"""

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import string
import ipdb
import numpy as np
from scipy.stats import norm
import sys
from itertools import combinations
from ucimlrepo import fetch_ucirepo

# Carregar conjunto de dados
idDS = 105  # votação congressional

try:
    ds = fetch_ucirepo(id=idDS)
    X = ds.data.features
    old_columns = X.columns.tolist()
    X.columns = [f'x{i+1}' for i in range(X.shape[1])]

    # Renomear colunas
    new_columns = X.columns.tolist()
    for old, new in zip(old_columns, new_columns):
        print(f"{old} -> {new}")
    y = ds.data.targets
    target = y.columns[0]
    attrs = X.columns

    # Verificar valores ausentes
    print(X.isna().sum())
    missing_percentage = (X.isna().sum().sum() / X.size) * 100
    print(f"Percentage of missing values: {missing_percentage:.2f}%")

except Exception as e:
    print(f"Error fetching or processing the dataset: {e}")
    print("Please check your internet connection and the availability of dataset ID 105 on the UC Irvine ML Repository.")
    X = None  # ou um DataFrame vazio: X = pd.DataFrame()

    # Lidar com o caso onde o carregamento de dados falhou (por exemplo, sair ou usar um conjunto de dados padrão)
    sys.exit() # terminar script se o carregamento for essencial

print(ds.data.features)

# Criar uma cópia do DataFrame original
X_copy = X.copy()
# Preencher valores ausentes com a moda de cada coluna
X_copy.fillna(X.mode().iloc[0], inplace=True)

# Função para calcular probabilidades básicas para combinações de atributos
def computeBasicProb(combination, operator):
    nRow = X_copy.shape[0]
    probabilities = []

    if not combination:
        return probabilities

    if operator == "and":
        contagem_combinacoes = X_copy.groupby(combination).size().reset_index(name='Contagem').dropna()
        for _, row in contagem_combinacoes.iterrows():
            probString = "P(" + ", ".join(f"{col}=={row[col]}" for col in combination) + f") >= {row['Contagem'] / nRow}"
            probabilities.append(probString)

    elif operator == "or":
        # Calcular corretamente a probabilidade 'or'.
        total_rows = len(X_copy)
        present_in_any = X_copy[combination].notna().any(axis=1).sum()  # Contar linhas onde pelo menos um atributo não é NaN
        prob_final = present_in_any / total_rows

        probString = "P(" + " or ".join(combination) + f") >= {prob_final}"
        probabilities.append(probString)

    return probabilities

# Gerar combinações de atributos
maxComprCombinacoes = int(input("Informe a quantidade máxima de variáveis em uma expressão básica and/or: "))  # Valor padrão para o número máximo de variáveis em uma expressão básica (and/or)
all_combinations = [list(comb) for i in range(1, maxComprCombinacoes + 1) for comb in combinations(attrs, i)]

probabilities = []
for combination in all_combinations:
    jointProbs = computeBasicProb(combination, "and")
    probabilities.extend(jointProbs)

    if len(combination) > 1:
        disjointProbs = computeBasicProb(combination, "or")
        probabilities.extend(disjointProbs)

# Salvar probabilidades no arquivo
with open("probabilidades.txt", "w") as f:
    for prob in probabilities:
        f.write(prob + "\n")

# Transformar os dados para o formato binário (True/False)
X_binario = X.map(lambda x: x == 'y')

# Aplicar o algoritmo Apriori
freq_items = apriori(X_binario, min_support=0.3, use_colnames=True)

# Gerar as regras de associação
regras = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# Salvar regras no arquivo
with open("apriori_regras.txt", "w") as f:
     f.write(regras.to_string())

# Função para calcular intervalos de confiança de Wald para probabilidades
def compute_wald_confidence_interval(prob, n, confidence=0.95):
  Z = norm.ppf((1 + confidence) / 2)  # Valor crítico para o nível de confiança
  margin_error = Z * np.sqrt((prob * (1 - prob)) / n)
  return max(0, prob - margin_error), min(1, prob + margin_error)

# Calcular intervalos de confiança para cada probabilidade
confidence_intervals = []
n = X.shape[0]  # Tamanho total da amostra

for prob in probabilities:
    try:
        p_value = float(prob.split(">=")[1].strip())  # Extrair o valor de probabilidade
        lower, upper = compute_wald_confidence_interval(p_value, n)
        confidence_intervals.append(f"{prob}, IC(95%): [{lower:.4f}, {upper:.4f}]")
    except Exception as e:
        print(f"Erro ao calcular IC: {e}, Prob: {prob}")
        continue  # Se não puder converter, ignora

# Salvar intervalos de confiança no arquivo
with open("intervalos_confianca.txt", "w") as f:
    for interval in confidence_intervals:
        f.write(interval + "\n")

print("Intervalos de confianca salvos em 'intervalos_confianca.txt'")
print("Probabilidades salvas em 'probabilidades.txt'")

#Solicite ao usuário que informe as probabilidades que deseja saber entre as variáveis por exemplo P(x1==y, x2==y)
while True:
    user_input = input("Informe as variáveis para calcular a probabilidade (Ex.: x1, x2 ou x1 || x2) ou 'sair' para terminar: ")
    if user_input.lower() == 'sair':
        break
    try:
        if "||" in user_input:
            operator = "or"
            variables = [var.strip() for var in user_input.split('||')]
        elif "," in user_input:
            operator = "and"
            variables = [var.strip() for var in user_input.split(',')]
        else:
            print("Formato inválido. Use ',' para 'and' ou '||' para 'or'.")
            continue
        
        print(f"variables: {variables}")
        prob_result = computeBasicProb(variables, operator)
        if prob_result:
            print("Probabilidades calculadas:")
            for prob in prob_result:
                print(prob)
        else:
            print("Não foi possível calcular a probabilidade para as variáveis fornecidas.")
    except Exception as e:
        print(f"Erro ao calcular a probabilidade: {e}")