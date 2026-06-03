# Relatorio do Modelo

## Dataset

O projeto usa um dataset sintetico e reproduzivel de telemetria de satelite gerado por `generate_dataset.py`.
O dataset tem 1000 linhas e 11 colunas: 10 variaveis de entrada mais o alvo binario `component_failure`.

- Distribuicao do alvo: {0: 641, 1: 359}
- Valores ausentes: 0
- Tipo do problema: classificacao binaria, em que `1` indica falha do componente.

## Pipeline

O pipeline de treinamento usa `SatelliteFeatureEngineer`, `StandardScaler` e um classificador supervisionado.
As variaveis criadas por engenharia de atributos sao:

- `thermal_stress`: distancia absoluta da temperatura nominal de 25 C.
- `low_voltage_risk`: quanto a bateria esta abaixo de 12 V.
- `fuel_reserve_risk`: quanto o combustivel esta abaixo de 500 kg.
- `radiation_age_interaction`: proxy de exposicao acumulada usando radiacao vezes tempo operacional.
- `orientation_magnitude`: magnitude do vetor de orientacao.

Os dados sao divididos com estratificacao em 80% treino e 20% teste. O conjunto de treino tambem e avaliado com validacao cruzada estratificada de 5 folds.

## Modelos Comparados

| model | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc | cv_f1 | cv_roc_auc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.635 | 0.494 | 0.528 | 0.510 | 0.639 | 0.549 | 0.686 |
| Random Forest | 0.595 | 0.418 | 0.319 | 0.362 | 0.607 | 0.462 | 0.659 |
| Gradient Boosting | 0.580 | 0.333 | 0.167 | 0.222 | 0.555 | 0.378 | 0.641 |

## Melhor Modelo

O modelo selecionado e **Logistic Regression**, escolhido pelo maior F1-score no teste. F1 e a metrica principal porque a classe positiva representa falha do componente e e menos frequente que a classe normal.

Matriz de confusao do modelo selecionado (`[[TN, FP], [FN, TP]]`):

```text
[[89, 39], [34, 38]]
```

## Criterio de alerta no app

No Streamlit, a previsao e exibida como classificacao operacional de risco usando a probabilidade estimada pelo modelo. O alerta comeca em 40% para priorizar a deteccao de cenarios suspeitos: abaixo de 40% e baixo risco, de 40% a 59,9% e risco moderado, e a partir de 60% e alto risco.

Esse limiar afeta apenas a leitura operacional no app. As metricas de validacao do modelo foram calculadas e registradas separadamente em `model_metrics.csv`.

## Interpretabilidade com SHAP

O SHAP foi calculado para o modelo selecionado depois das mesmas etapas de engenharia de atributos e normalizacao usadas no treinamento. Os maiores valores medios absolutos indicam as variaveis que mais influenciaram as previsoes de forma global.

| feature | mean_abs_shap |
| --- | ---: |
| age_years | 0.5478 |
| cosmic_radiation | 0.5048 |
| low_voltage_risk | 0.4266 |
| radiation_age_interaction | 0.3136 |
| thermal_stress | 0.2178 |
| data_rate_mbps | 0.1425 |
| thruster_fuel_kg | 0.1403 |
| orientation_magnitude | 0.1373 |
| fuel_reserve_risk | 0.1366 |
| orientation_z | 0.1032 |

## Limitacoes

O dataset e sintetico e deve ser tratado como simulacao, nao como telemetria certificada de missao real. Um uso real exigiria dados observados de satelites, validacao operacional, monitoramento de drift e revisao por especialistas de missao.

## Reprodutibilidade

```powershell
pip install -r requirements.txt
python generate_dataset.py
python train_save_model.py
python -m streamlit run app.py
```


