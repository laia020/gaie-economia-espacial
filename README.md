Generative AI for Engineering (GAIE) — Project Space Economy
Contexto do Problema

A nova corrida espacial é uma corrida de software. Observatórios, satélites e missões
científicas geram enormes volumes de dados orbitais sobre temperatura, radiação,
orientação, fluxo solar e outros parâmetros que influenciam a saúde de
componentes eletrônicos em órbita. Se um componente crítico de um satélite
falha, a missão pode ficar comprometida e o investimento perdido. O desafio
consiste em aproveitar técnicas de Inteligência Artificial e Machine Learning
para analisar telemetria e prever a probabilidade de falha de componentes,
melhorando a resiliência das missões e impulsionando a economia espacial.

A documentação oficial da NASA descreve que o serviço NeoWs fornece acesso a
informações sobre objetos próximos à Terra, incluindo magnitude absoluta, dados
de aproximação, distância e velocidade relativa. Embora não
seja utilizado diretamente aqui, essa referência inspirou a seleção de
variáveis que refletem medições reais de missões espaciais. Com base na
telemetria de satélites (temperatura, voltagem da bateria, radiação cósmica,
fluxo solar, combustível do propulsor etc.), criamos um conjunto de dados
sintético para treinar e testar modelos preditivos.

Conjunto de Dados

Foi gerado um dataset sintético com 1 000 linhas e 10 colunas, simulando
leituras de telemetria de um satélite:

Variável	Descrição
temperature_c	Temperatura do componente em °C.
battery_voltage	Voltagem da bateria em volts.
orientation_x/y/z	Componentes de orientação (vetor unitário).
cosmic_radiation	Contagens de radiação cósmica.
solar_flux	Fluxo solar incidente em W/m².
data_rate_mbps	Taxa de transmissão de dados em Mbps.
thruster_fuel_kg	Combustível restante no propulsor em kg.
age_years	Idade do satélite em anos.
component_failure	Variável alvo binária (1=falha, 0=normal).

A coluna component_failure foi calculada a partir de uma função logística que
relaciona radiação, temperatura, voltagem e idade à probabilidade de falha. O
dataset completo está disponível em satellite_failure_dataset.csv.

Metodologia
Preparação dos dados — Os dados foram divididos em conjuntos de
treinamento e teste (80/20). Como todas as variáveis são numéricas,
aplicou‑se normalização (StandardScaler) para padronizar médias e desvio
padrão.
Modelos preditivos — Foram treinados dois modelos supervisionados:
Regressão Logística — Pipeline com normalização + LogisticRegression.
Random Forest — Pipeline com normalização + RandomForestClassifier.
Validação e métricas — Avaliou‑se acurácia, precisão, recall e F1‑score
usando os dados de teste. A Regressão Logística apresentou melhor F1‑score
(≈0,31) do que a Random Forest (≈0,24). Portanto, a Regressão Logística foi
selecionada para deploy. Detalhes completos das métricas constam em
model_report.md.
Interpretabilidade — Utilizou‑se SHAP para explicar as previsões do
modelo selecionado. Foi calculado o valor absoluto médio dos SHAP values
para cada variável, mostrando que combustível do propulsor e radiação
cósmica são as variáveis que mais influenciam a previsão de falha, seguidas
pela taxa de dados e temperatura.

Deploy da aplicação — Foi desenvolvida uma aplicação Web com Streamlit
(app.py). Ela permite ao usuário ajustar valores das variáveis de
telemetria, predizer a probabilidade de falha e visualizar um gráfico de
SHAP que explica a decisão. Para rodar localmente:

pip install -r requirements.txt
streamlit run app.py

A pasta contém ainda o arquivo logistic_model.pkl com o pipeline
treinado e satellite_failure_dataset.csv com os dados para consulta.

Resultados Obtidos
Regressão Logística – F1 score ≈ 0,31; acurácia ≈ 0,65; precisão ≈ 0,48;
recall ≈ 0,23.
Random Forest – F1 score ≈ 0,24; acurácia ≈ 0,65; precisão ≈ 0,48;
recall ≈ 0,16.
Variáveis mais importantes (SHAP): thruster_fuel_kg, cosmic_radiation,
data_rate_mbps, temperature_c e age_years.

Esses resultados indicam que, mesmo em um cenário sintético, é possível
identificar fatores relevantes para a falha de componentes com um modelo
simples. Em aplicações reais, dados de missões espaciais e APIs como a
NeoWs (que fornece magnitude absoluta, dados de aproximação, distância e
velocidade para objetos próximos à Terra) podem ser
integrados para enriquecer o conjunto de características e melhorar o
modelo.

Instruções para Execução do Projeto

Clone este repositório:

git clone <URL_DO_REPOSITORIO>
cd <diretorio>

Instale as dependências:

pip install -r requirements.txt

Execute o script app.py com o Streamlit:

streamlit run app.py
Ajuste os controles deslizantes e campos numéricos para ver a
probabilidade de falha e a explicação das previsões.
Organização do Repositório
satellite_failure_dataset.csv – Dataset sintético de telemetria.
train_save_model.py – Script para treinar e salvar o modelo.
logistic_model.pkl – Pipeline de Regressão Logística treinado.
model_report.md – Relatório resumindo métricas e importância das variáveis.
app.py – Aplicação Streamlit para interação com o modelo e visualização de SHAP.
README.md – Este documento.
Próximos Passos
Coleta de dados reais – Utilizar APIs públicas, como a NeoWs ou
outras fontes de telemetria espacial, para substituir o dataset sintético por
informações observadas. A API NeoWs permite navegar pelo dataset de
asteroides, procurando por objetos com base em sua data de aproximação,
buscar um asteroide específico ou navegar pelo conjunto completo.
Explorar modelos mais avançados – Testar técnicas de
floresta gradiente, redes neurais e autoencoders de detecção de anomalias.
Monitoramento em tempo real – Integrar a solução com fluxos de dados
em tempo real e alertar engenheiros quando a probabilidade de falha
ultrapassar um limiar definido.

Este projeto demonstra um pipeline completo de IA/ML aplicado à economia
espacial, desde a geração de dados, treinamento de modelos, validação,
interpretação com SHAP até o deploy de uma aplicação interativa. Ajuste,
teste e expanda conforme sua criatividade e as necessidades do desafio.
