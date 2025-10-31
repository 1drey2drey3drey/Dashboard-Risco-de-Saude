# 🩺 Projeto de Análise e Dashboard Interativo de Risco de Saúde

## 📋 Descrição

Este projeto é uma aplicação web interativa desenvolvida em **Dash** (Python) para a análise e visualização de dados de saúde, com foco na avaliação de fatores de risco.

O Dashboard permite explorar o conjunto de dados, visualizar as estatísticas principais e aplicar modelos preditivos e testes estatísticos diretamente na interface.

## 💾 Dados

O projeto utiliza o arquivo `Health_Risk_Dataset.csv`, que contém informações de sinais vitais de pacientes e um nível de risco de saúde associado.

As colunas principais no dataset incluem:

* `Respiratory_Rate` (Frequência Respiratória)
* `Oxygen_Saturation` (Saturação de Oxigênio)
* `Systolic_BP` (Pressão Arterial Sistólica)
* `Heart_Rate` (Frequência Cardíaca)
* `Temperature` (Temperatura)
* `Consciousness` (Nível de Consciência)
* `Risk_Level` (Nível de Risco: Normal, Baixo, Médio, Alto)

## ✨ Funcionalidades

O código do `app.py` implementa as seguintes funcionalidades principais no Dashboard:

1. **Pré-processamento e Padronização:**
   * Criação de uma nova variável binária, `Febre`, baseada na `Temperature` (> 37.5°C)
   * Mapeamento e tradução dos Níveis de Risco (`Risk_Level`) e Níveis de Consciência (`Consciousness`) para o Português

2. **Dashboard Interativo (Dash/Plotly):**
   * Interface de usuário com temas claro/escuro (Bootstrap/DBC)
   * Visualizações interativas de dados (gráficos de distribuição, dispersão, etc.)

3. **Análise Estatística:**
   * **Intervalo de Confiança (IC 95%):** Ferramenta para calcular e exibir o Intervalo de Confiança de 95% para uma variável numérica selecionada
   * **Cálculo de p-value:** Ferramenta para calcular o p-value associado a uma coluna selecionada

4. **Modelagem Preditiva (Regressão Linear):**
   * Recurso para executar um modelo de Regressão Linear com variáveis de entrada e saída selecionáveis
   * Exibição dos resultados do modelo, incluindo:
     * Coeficientes de regressão (indicando o impacto de cada variável)
     * Métricas de desempenho: **R²** (Coeficiente de Determinação) e **RMSE** (Erro Quadrático Médio da Raiz)

## ⚙️ Estrutura do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `app.py` | Contém o código completo da aplicação web interativa **Dash**, definindo o layout e todas as callbacks de funcionalidade (gráficos, regressão linear, IC, p-value) |
| `analise.ipynb` | Notebook Jupyter que documenta a **Análise Exploratória de Dados (EDA)**, visualizações iniciais e a experimentação da **Regressão Linear** em diferentes variáveis de saúde (`Heart_Rate`, `Respiratory_Rate`, `Systolic_BP`) |
| `Health_Risk_Dataset.csv` | O conjunto de dados de saúde utilizado para análise e construção do Dashboard |

## 🛠️ Tecnologias Utilizadas

* **Python**
* **Dash**
* **Plotly Express**
* **Pandas**
* **Scikit-learn** (`LinearRegression`, `mean_squared_error`, `r2_score`)
* **SciPy** (`stats.t.interval`, `stats.sem`)
* **Dash Bootstrap Components** (para estilização e temas)

## 🚀 Como Executar

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install dash plotly pandas scikit-learn scipy dash-bootstrap-components
   ```
3. Execute a aplicação:
   ```bash
   python app.py
   ```
4. Acesse o dashboard no navegador: `http://localhost:8050`

## 📊 Exemplos de Uso

* Explore as distribuições dos sinais vitais por nível de risco
* Calcule intervalos de confiança para variáveis críticas
* Treine modelos preditivos para identificar fatores de risco
* Compare estatísticas entre diferentes grupos de pacientes