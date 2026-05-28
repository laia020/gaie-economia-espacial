# GAIE — Inteligência Artificial aplicada à Economia Espacial

Projeto desenvolvido para a disciplina **Generative AI For Engineering (GAIE)**, como parte da **Global Solution FIAP — Space Connect**.

A proposta do projeto é construir um pipeline completo de **Inteligência Artificial e Machine Learning** aplicado a um problema real relacionado à **Economia Espacial**.

---

## Aplicação em funcionamento

A aplicação foi publicada utilizando **Streamlit Community Cloud** e pode ser acessada pelo link abaixo:

https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

---

## Repositório do projeto

https://github.com/laia020/gaie-economia-espacial

---

## Contexto do problema

A nova corrida espacial não depende apenas de foguetes, satélites e estações orbitais. Ela depende cada vez mais de **software, dados, automação e inteligência artificial**.

Satélites modernos operam em ambientes extremos, sujeitos a variações de temperatura, radiação cósmica, falhas de energia, perda de comunicação, degradação de componentes e limitações de combustível. Uma falha em um componente crítico pode comprometer a missão, gerar prejuízos financeiros e afetar serviços essenciais na Terra, como telecomunicações, monitoramento climático, navegação, agricultura de precisão e resposta a desastres.

Diante desse cenário, este projeto propõe uma solução de IA para **prever falhas em componentes de satélites** a partir de dados simulados de telemetria.

---

## Objetivo do projeto

O objetivo é desenvolver um pipeline completo de Machine Learning capaz de prever se um componente de satélite apresenta risco de falha, com base em variáveis operacionais e ambientais.

A solução contempla:

- Geração de dados sintéticos com mais de 1.000 registros.
- Pré-processamento dos dados.
- Treinamento de modelos preditivos.
- Comparação de desempenho entre modelos.
- Escolha do melhor modelo.
- Interpretação das previsões com SHAP.
- Deploy da solução em uma aplicação web com Streamlit.
- Documentação completa para reprodução do projeto.

---

## Problema de Machine Learning

Este projeto trata um problema de **classificação binária**.

A variável alvo é:

| Variável | Descrição |
|---|---|
| `component_failure` | Indica se houve falha no componente do satélite. `0 = sem falha`, `1 = falha` |

O modelo recebe dados de telemetria como entrada e retorna a probabilidade de falha.

---

## Fonte dos dados

Foi utilizado um **dataset sintético gerado por IA**, seguindo a exigência da entrega de conter no mínimo:

- 1.000 linhas.
- 10 colunas.
- Variáveis relacionadas ao problema escolhido.

O dataset simula dados de telemetria de satélites, considerando características comuns em ambientes orbitais.

Arquivo utilizado:

```txt
satellite_failure_dataset.csv
```

---

## Dicionário de dados

| Coluna | Descrição |
|---|---|
| `temperature_c` | Temperatura do componente em graus Celsius. |
| `battery_voltage` | Tensão da bateria do satélite. |
| `orientation_x` | Eixo X da orientação do satélite. |
| `orientation_y` | Eixo Y da orientação do satélite. |
| `orientation_z` | Eixo Z da orientação do satélite. |
| `cosmic_radiation` | Nível de radiação cósmica recebido. |
| `solar_flux` | Intensidade do fluxo solar. |
| `data_rate_mbps` | Taxa de transmissão de dados em Mbps. |
| `thruster_fuel_kg` | Quantidade de combustível restante nos propulsores. |
| `age_years` | Idade operacional do satélite em anos. |
| `component_failure` | Variável alvo: indica se houve falha no componente. |

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- GitHub

---

## Estrutura do projeto

```txt
gaie-economia-espacial/
│
├── app.py
├── train_save_model.py
├── satellite_failure_dataset.csv
├── logistic_model.pkl
├── model_report.md
├── requirements.txt
└── README.md
```

---

## Pipeline de Machine Learning

O pipeline desenvolvido contempla as seguintes etapas:

### 1. Geração dos dados

Foi criado um dataset sintético simulando dados de telemetria de satélites.

As variáveis foram definidas com base em fatores que podem influenciar falhas em componentes espaciais, como temperatura, radiação, energia, idade do satélite e combustível restante.

---

### 2. Pré-processamento

As variáveis numéricas foram padronizadas utilizando `StandardScaler`, garantindo que os modelos trabalhassem com dados em uma escala adequada.

Também foi feita a separação entre:

- Variáveis preditoras.
- Variável alvo.
- Base de treino.
- Base de teste.

---

### 3. Modelos testados

Foram aplicados dois modelos de classificação:

#### Regressão Logística

Modelo linear utilizado como baseline para classificação binária.

#### Random Forest Classifier

Modelo baseado em árvores de decisão, utilizado para capturar relações não lineares entre as variáveis.

---

### 4. Validação dos modelos

Os modelos foram avaliados utilizando as seguintes métricas:

- Accuracy.
- Precision.
- Recall.
- F1-score.

---

## Resultados obtidos

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.645 | 0.485 | 0.229 | 0.311 |
| Random Forest | 0.645 | 0.478 | 0.157 | 0.237 |

Com base no F1-score, o modelo escolhido foi:

```txt
Logistic Regression
```

A Regressão Logística apresentou melhor equilíbrio entre precisão e recall neste conjunto de dados.

---

## Interpretabilidade com SHAP

Para interpretar o comportamento do modelo, foi utilizada a biblioteca **SHAP**.

O SHAP permite identificar quais variáveis mais influenciaram as previsões do modelo, aumentando a transparência da solução.

As variáveis com maior influência foram:

| Variável | Influência média SHAP |
|---|---:|
| `thruster_fuel_kg` | 9.927876 |
| `cosmic_radiation` | 5.191791 |
| `data_rate_mbps` | 1.354582 |
| `temperature_c` | 0.825029 |
| `age_years` | 0.807272 |
| `solar_flux` | 0.600227 |
| `battery_voltage` | 0.175467 |
| `orientation_y` | 0.081065 |
| `orientation_z` | 0.045517 |
| `orientation_x` | 0.035726 |

A análise mostra que fatores como combustível restante, radiação cósmica, taxa de transmissão, temperatura e idade do satélite possuem forte impacto na previsão de falha.

---

## Aplicação web

A aplicação foi desenvolvida com **Streamlit**.

Ela permite que o usuário informe valores de telemetria e receba:

- Probabilidade de falha do componente.
- Classificação final: falha ou sem falha.
- Explicação da decisão do modelo com SHAP.

Arquivo principal da aplicação:

```txt
app.py
```

---

## Como executar o projeto localmente

### 1. Clonar o repositório

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

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar a aplicação

```bash
streamlit run app.py
```

Depois disso, a aplicação será aberta no navegador.

---

## Como treinar o modelo novamente

Caso deseje treinar o modelo novamente, execute:

```bash
python train_save_model.py
```

Esse script irá:

1. Carregar o dataset.
2. Separar treino e teste.
3. Aplicar o pipeline de pré-processamento.
4. Treinar o modelo.
5. Salvar o modelo treinado em:

```txt
logistic_model.pkl
```

---

## Deploy

O deploy foi realizado utilizando **Streamlit Community Cloud**.

Link da aplicação:

https://gaie-economia-espacial-hmh597lhuamwgbmtqwza9j.streamlit.app/

---

## Critérios da entrega atendidos

| Critério | Status |
|---|---|
| Definição de problema relacionado à Economia Espacial | Concluído |
| Dataset com no mínimo 1.000 linhas | Concluído |
| Dataset com no mínimo 10 colunas | Concluído |
| Desenvolvimento de modelos preditivos | Concluído |
| Aplicação de pelo menos duas técnicas de ML | Concluído |
| Pipeline com pré-processamento | Concluído |
| Treinamento dos modelos | Concluído |
| Validação e comparação de desempenho | Concluído |
| Escolha do melhor modelo | Concluído |
| Interpretabilidade com SHAP | Concluído |
| Deploy da aplicação | Concluído |
| README detalhado | Concluído |
| Código disponível no GitHub | Concluído |

---

## Conexão com a Economia Espacial

A solução se conecta diretamente à Economia Espacial por atuar sobre um problema crítico de operação de satélites: a previsão de falhas em componentes.

Satélites são ativos essenciais para diversos setores econômicos, como:

- Telecomunicações.
- Agricultura de precisão.
- Monitoramento ambiental.
- Defesa.
- Navegação.
- Clima e meteorologia.
- Internet via satélite.
- Observação da Terra.

A capacidade de prever falhas permite reduzir custos, aumentar a vida útil de missões e melhorar a confiabilidade de serviços baseados em infraestrutura espacial.

---

## Possíveis melhorias futuras

- Utilizar dados reais de telemetria espacial.
- Integrar APIs públicas da NASA, ESA, INPE ou Copernicus.
- Testar modelos mais avançados, como XGBoost, LightGBM e redes neurais.
- Criar um sistema de alerta em tempo real.
- Armazenar previsões em banco de dados.
- Adicionar autenticação de usuários.
- Criar dashboard com histórico de falhas.
- Melhorar o balanceamento das classes do dataset.
- Aplicar técnicas de detecção de anomalias.

---

## Autores

Projeto desenvolvido para a Global Solution FIAP — Generative AI For Engineering.

Equipe:

```txt
Nome dos integrantes aqui
RM dos integrantes aqui
```

---

## Licença

Este projeto foi desenvolvido para fins acadêmicos.
