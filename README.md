\# 🩺 Projeto de Análise e Dashboard Interativo de Risco de Saúde



\## 📋 Descrição



Este projeto é uma aplicação web interativa desenvolvida em \*\*Dash\*\* (Python) para a análise e visualização de dados de saúde, com foco na avaliação de fatores de risco.



O Dashboard permite explorar o conjunto de dados, visualizar as estatísticas principais e aplicar modelos preditivos e testes estatísticos diretamente na interface.



\## 💾 Dados



O projeto utiliza o arquivo `Health\_Risk\_Dataset.csv`, que contém informações de sinais vitais de pacientes e um nível de risco de saúde associado.



As colunas principais no \*dataset\* incluem:

\* `Respiratory\_Rate` (Frequência Respiratória)

\* `Oxygen\_Saturation` (Saturação de Oxigénio)

\* `Systolic\_BP` (Pressão Arterial Sistólica)

\* `Heart\_Rate` (Frequência Cardíaca)

\* `Temperature` (Temperatura)

\* `Consciousness` (Nível de Consciência)

\* `Risk\_Level` (Nível de Risco: Normal, Baixo, Médio, Alto)



\## ✨ Funcionalidades



O código do `app.py` implementa as seguintes funcionalidades principais no Dashboard:



1\.  \*\*Pré-processamento e Padronização:\*\*

&nbsp;   \* Criação de uma nova variável binária, `Febre`, baseada na `Temperature` (> 37.5).

&nbsp;   \* Mapeamento e tradução dos Níveis de Risco (`Risk\_Level`) e Níveis de Consciência (`Consciousness`) para o Português.

2\.  \*\*Dashboard Interativo (Dash/Plotly):\*\*

&nbsp;   \* Interface de usuário com temas claro/escuro (Bootstrap/DBC).

&nbsp;   \* Visualizações interativas de dados (gráficos de distribuição, dispersão, etc., que podem ser selecionadas e exibidas no Dashboard - \*implicado pelo uso do Plotly e Dash\*).

3\.  \*\*Análise Estatística:\*\*

&nbsp;   \* \*\*Intervalo de Confiança (IC 95%):\*\* Ferramenta para calcular e exibir o Intervalo de Confiança de 95% para uma variável numérica selecionada.

&nbsp;   \* \*\*Cálculo de p-value:\*\* Ferramenta para calcular o p-value associado a uma coluna selecionada.

4\.  \*\*Modelagem Preditiva (Regressão Linear):\*\*

&nbsp;   \* Recurso para executar um modelo de Regressão Linear com variáveis de entrada e saída selecionáveis.

&nbsp;   \* Exibição dos resultados do modelo, incluindo:

&nbsp;       \* Coeficientes de regressão (indicando o impacto de cada variável).

&nbsp;       \* Métricas de desempenho: \*\*$R^2$\*\* (Coeficiente de Determinação) e \*\*RMSE\*\* (Erro Quadrático Médio da Raiz).



\## ⚙️ Estrutura do Projeto



| Arquivo | Descrição |

| :--- | :--- |

| `app.py` | Contém o código completo da aplicação web interativa \*\*Dash\*\*, definindo o \*layout\* e todas as \*callbacks\* de funcionalidade (gráficos, regressão linear, IC, p-value). |

| `analise.ipynb` | Notebook Jupyter que documenta a \*\*Análise Exploratória de Dados (EDA)\*\*, visualizações iniciais e a experimentação da \*\*Regressão Linear\*\* em diferentes variáveis de saúde (`Heart\_Rate`, `Respiratory\_Rate`, `Systolic\_BP`). |

| `Health\_Risk\_Dataset.csv` | O conjunto de dados de saúde utilizado para análise e construção do Dashboard. |



\## 🛠️ Tecnologias Utilizadas



\* \*\*Python\*\*

\* \*\*Dash\*\*

\* \*\*Plotly Express\*\*

\* \*\*Pandas\*\*

\* \*\*Scikit-learn\*\* (`LinearRegression`, `mean\_squared\_error`, `r2\_score`)

\* \*\*SciPy\*\* (`stats.t.interval`, `stats.sem`)

\* \*\*Dash Bootstrap Components\*\* (para estilização e temas)

