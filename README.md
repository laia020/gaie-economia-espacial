# GAIE - Inteligencia Artificial aplicada a Economia Espacial

Projeto desenvolvido para a disciplina **Generative AI For Engineering (GAIE)**, como parte da **Global Solution FIAP - Space Connect**.

A proposta e construir um pipeline completo de **Inteligencia Artificial e Machine Learning** aplicado a um problema de **Economia Espacial**: prever falhas em componentes de satelites a partir de dados de telemetria.

## Aplicacao em funcionamento

A aplicacao foi publicada com **Streamlit Community Cloud**:

https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

## Repositorio do projeto

https://github.com/laia020/gaie-economia-espacial

## Contexto do problema

A nova corrida espacial depende cada vez mais de software, dados, automacao e inteligencia artificial. Satelites modernos operam em ambientes extremos, sujeitos a variacoes de temperatura, radiacao cosmica, limitacoes de energia, degradacao de componentes e consumo de combustivel.

Uma falha em um componente critico pode comprometer a missao, gerar prejuizos financeiros e afetar servicos essenciais na Terra, como telecomunicacoes, monitoramento ambiental, navegacao, agricultura de precisao e resposta a desastres.

Este projeto propoe uma solucao de IA para estimar a probabilidade de falha de componentes de satelites com base em leituras simuladas de telemetria.

## Objetivo

Desenvolver, treinar, comparar, interpretar e publicar uma solucao de Machine Learning capaz de prever se um componente de satelite apresenta risco de falha.

A solucao contempla:

- Geracao de dataset sintetico reproduzivel.
- Pre-processamento e engenharia de atributos.
- Treinamento e comparacao de modelos preditivos.
- Escolha do melhor modelo com base em F1-score.
- Interpretabilidade com SHAP.
- Deploy em aplicacao web com Streamlit.
- Documentacao e instrucoes de reproducao.

## Problema de Machine Learning

O projeto trata um problema de **classificacao binaria**.

| Variavel alvo | Descricao |
|---|---|
| `component_failure` | Indica falha do componente. `0 = sem falha`, `1 = falha`. |

O modelo recebe variaveis de telemetria e retorna a probabilidade de falha.

## Fonte e geracao dos dados

O dataset e sintetico e reproduzivel. Ele e gerado pelo script:

```txt
generate_dataset.py
```

O arquivo gerado e:

```txt
satellite_failure_dataset.csv
```

Caracteristicas do dataset:

- 1000 linhas.
- 10 variaveis preditoras.
- 1 variavel alvo.
- 0 valores ausentes.
- Distribuicao atual do alvo: 641 registros sem falha e 359 registros com falha.

## Dicionario de dados

| Coluna | Descricao |
|---|---|
| `temperature_c` | Temperatura do componente em graus Celsius. |
| `battery_voltage` | Tensao da bateria do satelite. |
| `orientation_x` | Eixo X da orientacao do satelite. |
| `orientation_y` | Eixo Y da orientacao do satelite. |
| `orientation_z` | Eixo Z da orientacao do satelite. |
| `cosmic_radiation` | Nivel de radiacao cosmica recebido. |
| `solar_flux` | Intensidade do fluxo solar. |
| `data_rate_mbps` | Taxa de transmissao de dados em Mbps. |
| `thruster_fuel_kg` | Quantidade de combustivel restante nos propulsores. |
| `age_years` | Idade operacional do satelite em anos. |
| `component_failure` | Variavel alvo: indica se houve falha no componente. |

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- GitHub

## Estrutura do projeto

```txt
gaie-economia-espacial/
|-- app.py
|-- best_model.pkl
|-- feature_engineering.py
|-- generate_dataset.py
|-- logistic_model.pkl
|-- model_metadata.json
|-- model_metrics.csv
|-- model_report.md
|-- requirements.txt
|-- satellite_failure_dataset.csv
|-- shap_summary.csv
|-- train_save_model.py
|-- docs/
|   |-- advanced-ml-improvements.md
|   `-- project-deliverables.md
`-- README.md
```

## Pipeline de Machine Learning

### 1. Geracao dos dados

O script `generate_dataset.py` cria dados sinteticos com seed fixa. As variaveis simulam condicoes de operacao de satelites, como temperatura, radiacao, tensao da bateria, combustivel, fluxo solar, taxa de dados e idade.

A variavel `component_failure` e calculada a partir de uma funcao logistica que combina fatores de risco.

### 2. Engenharia de atributos

O arquivo `feature_engineering.py` adiciona variaveis derivadas ao pipeline:

| Feature criada | Ideia |
|---|---|
| `thermal_stress` | Distancia da temperatura nominal de 25 C. |
| `low_voltage_risk` | Risco por tensao abaixo de 12 V. |
| `fuel_reserve_risk` | Risco por combustivel abaixo de 500 kg. |
| `radiation_age_interaction` | Exposicao acumulada aproximada por radiacao vezes idade. |
| `orientation_magnitude` | Magnitude do vetor de orientacao. |

### 3. Pre-processamento

O pipeline usa:

- `SatelliteFeatureEngineer` para engenharia de atributos.
- `StandardScaler` para padronizacao numerica.
- Classificador supervisionado.

Os dados sao separados em treino e teste com estratificacao, preservando a proporcao entre classes.

### 4. Modelos testados

Foram comparados tres modelos:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

O treinamento e a comparacao ficam em `train_save_model.py`.

### 5. Validacao dos modelos

Foram usadas as metricas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusao

Tambem foi aplicada validacao cruzada estratificada com 5 folds no conjunto de treino.

## Resultados obtidos

Resultados atuais do arquivo `model_metrics.csv`:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.635 | 0.494 | 0.528 | 0.510 | 0.639 |
| Random Forest | 0.595 | 0.418 | 0.319 | 0.362 | 0.607 |
| Gradient Boosting | 0.580 | 0.333 | 0.167 | 0.222 | 0.555 |

O melhor modelo foi:

```txt
Logistic Regression
```

Ele foi escolhido pelo maior F1-score no conjunto de teste. O F1-score foi usado como metrica principal porque a classe positiva representa falha do componente e e menos frequente que a classe normal.

O modelo final esta salvo em:

```txt
best_model.pkl
```

## Interpretabilidade com SHAP

O projeto usa SHAP para explicar as previsoes do modelo escolhido.

- `shap_summary.csv` contem a importancia global das variaveis.
- `model_report.md` documenta a interpretacao global.
- `app.py` calcula SHAP local para a previsao feita pelo usuario.

Principais variaveis na ultima execucao:

| Feature | Mean absolute SHAP |
|---|---:|
| `age_years` | 0.5478 |
| `cosmic_radiation` | 0.5048 |
| `low_voltage_risk` | 0.4266 |
| `radiation_age_interaction` | 0.3136 |
| `thermal_stress` | 0.2178 |

A interpretacao indica que idade do satelite, radiacao cosmica, risco de baixa tensao e exposicao acumulada estao entre os fatores mais relevantes para a previsao de falha.

## Aplicacao web

A aplicacao foi desenvolvida com **Streamlit** em `app.py`.

Ela permite que o usuario informe valores de telemetria e receba:

- Probabilidade de falha do componente.
- Classificacao final: falha ou sem falha.
- Explicacao local da decisao com SHAP.
- Tabela de metricas dos modelos.

Link da aplicacao publicada:

https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

## Como executar localmente

### 1. Clonar o repositorio

```bash
git clone https://github.com/laia020/gaie-economia-espacial.git
cd gaie-economia-espacial
```

### 2. Criar ambiente virtual

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Gerar o dataset

```bash
python generate_dataset.py
```

### 5. Treinar e comparar os modelos

```bash
python train_save_model.py
```

Esse comando gera ou atualiza:

- `best_model.pkl`
- `logistic_model.pkl`
- `model_metrics.csv`
- `model_report.md`
- `model_metadata.json`
- `shap_summary.csv`

### 6. Executar a aplicacao

```bash
streamlit run app.py
```

Depois disso, acesse:

```txt
http://localhost:8501
```

## Deploy

O deploy foi realizado no **Streamlit Community Cloud**.

Link da aplicacao:

https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

## Criterios da entrega atendidos

| Criterio | Onde aparece | Status |
|---|---|---|
| Problema relacionado a Economia Espacial | README e `model_report.md` | Concluido |
| Dataset com no minimo 1000 linhas | `generate_dataset.py` e `satellite_failure_dataset.csv` | Concluido |
| Dataset com no minimo 10 colunas | `satellite_failure_dataset.csv` | Concluido |
| Fonte ou geracao dos dados | `generate_dataset.py` | Concluido |
| Pelo menos duas tecnicas de ML | `train_save_model.py` | Concluido |
| Pipeline com pre-processamento | `train_save_model.py` | Concluido |
| Engenharia de atributos | `feature_engineering.py` | Concluido |
| Treinamento dos modelos | `train_save_model.py` | Concluido |
| Validacao e comparacao de desempenho | `model_metrics.csv` e `model_report.md` | Concluido |
| Escolha do melhor modelo | `best_model.pkl` e `model_metadata.json` | Concluido |
| Interpretabilidade com SHAP | `shap_summary.csv`, `model_report.md` e `app.py` | Concluido |
| Deploy da aplicacao | Streamlit Community Cloud | Concluido |
| Codigo disponivel no GitHub | Repositorio publicado | Concluido |
| README detalhado | `README.md` | Concluido |

## Conexao com a Economia Espacial

A solucao se conecta diretamente a Economia Espacial porque atua em um problema critico de operacao de satelites: a previsao de falhas em componentes.

Satelites sao ativos essenciais para telecomunicacoes, navegacao, monitoramento ambiental, agricultura de precisao, defesa, meteorologia, observacao da Terra e internet via satelite. Prever falhas pode reduzir custos, aumentar a vida util de missoes e melhorar a confiabilidade de servicos baseados em infraestrutura espacial.

## Possiveis melhorias futuras

- Utilizar dados reais de telemetria espacial.
- Integrar APIs publicas da NASA, ESA, INPE ou Copernicus.
- Testar modelos mais avancados, como XGBoost, LightGBM e redes neurais.
- Calibrar probabilidades e ajustar threshold para reduzir falsos negativos.
- Criar sistema de alerta em tempo real.
- Armazenar previsoes em banco de dados.
- Criar dashboard com historico de falhas.
- Aplicar tecnicas de deteccao de anomalias.

## Autores

Projeto desenvolvido para a Global Solution FIAP - Generative AI For Engineering.

Equipe:

```txt
Lucas Laia Manentti - RM 97709
Guilherme Faustino Vargas - RM 98278
Ryan Perez Pacheco - RM 98782
```

## Licenca

Este projeto foi desenvolvido para fins academicos.

## Resumo executivo complementar

Este projeto simula um cenario de manutencao preditiva para satelites. A solucao gera um dataset sintetico de telemetria espacial, treina e compara tres modelos de classificacao, escolhe o melhor modelo por F1-score, explica as previsoes com SHAP e publica uma interface interativa em Streamlit.

- Problema: prever risco de falha em componentes de satelites.
- Dados: dataset sintetico reproduzivel com 1000 linhas, 10 variaveis preditoras e 1 alvo binario.
- Modelos comparados: Logistic Regression, Random Forest e Gradient Boosting.
- Melhor modelo atual: Logistic Regression, com F1-score de 0.510 no teste.
- Aplicacao publicada: https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

## Como o dataset foi montado

O dataset foi montado de forma sintetica para simular telemetria de um satelite em operacao.
O script generate_dataset.py usa uma seed fixa, RANDOM_SEED = 42, para que qualquer pessoa consiga gerar novamente o mesmo conjunto de dados.

As variaveis simulam leituras esperadas de uma missao espacial, como temperatura, tensao da bateria, orientacao, radiacao, fluxo solar, taxa de dados, combustivel e idade operacional.
A variavel alvo component_failure nao foi sorteada de forma totalmente aleatoria.
Primeiro, o script calcula fatores de risco como estresse termico, baixa tensao, baixo combustivel, magnitude da orientacao e exposicao a radiacao ao longo do tempo.
Depois, esses fatores entram em uma funcao logistica que gera a probabilidade de falha; a classe final e sorteada com base nessa probabilidade.

