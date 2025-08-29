# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:04:45 2025

@author: Julio Cesar Soares Ribeiro
"""
#%%
#Importar as bibliotecas para analise

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from prophet.plot import add_changepoints_to_plot

#%% Leitura da base

df = pd.read_excel("mov_estoque.xlsx")

#%% Consumo mensal
df['Data'] = pd.to_datetime(df['Data'])

df_consumo_mensal = df.resample('ME', on='Data').sum()

df_consumo_mensal.head()

#%% Criar uma tabela só com média, mediana, desvio-padrão e porcentagem

# Soma
soma = df_consumo_mensal.sum().sort_values(ascending = False)
#média
media = df_consumo_mensal.mean().sort_values(ascending = False)
#desvio-padrao
desvio_pad = df_consumo_mensal.std().sort_values(ascending =False)
#criar o DataFrame
analise = pd.DataFrame({'Desvio-pad':desvio_pad,'Media':media, 'Qtde_m2': soma}).sort_values(by='Qtde_m2', ascending=False)

#primeiras linhas
analise.head()

#%% identificar os itens com mais movimentos

soma_total = analise['Qtde_m2'].sum()
analise['Qtde_m2_%'] = ((analise['Qtde_m2']/soma_total)*100).round(2)
analise['Qtde_m2_acum_%'] = (analise['Qtde_m2_%'].cumsum()).round(2)

analise.head()
#%% Classificação ABC
# Defina as condições com base em Soma_acumulada_%
condicoes = [
    analise['Qtde_m2_acum_%'] <= 80,
    (analise['Qtde_m2_acum_%'] > 80) & (analise['Qtde_m2_acum_%'] <= 95),
    analise['Qtde_m2_acum_%'] > 95
]

# Rótulos correspondentes
categorias = ['A', 'B', 'C']

# Criação da nova coluna com a classificação ABC
analise['Classificação_ABC'] = np.select(condicoes, categorias, default='')

#calcular a distribuição da classificação
distribuicao = analise['Classificação_ABC'].value_counts(normalize=True).sort_index() * 100

#gráfico da distribuição itens por classe
plt.figure(figsize=(6, 4))
bars = distribuicao.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])

plt.title('Distribuição de Itens por Classe ABC')
plt.xlabel('Classe')
plt.ylabel('Porcentagem (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

# Adiciona os rótulos (porcentagens) no topo de cada barra
for i, valor in enumerate(distribuicao):
    plt.text(i, valor + 1, f'{valor:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

#gráfico da distribuição quantidade por classe
analise.groupby('Classificação_ABC').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

plt.title('Distribuição de Itens por Classe ABC')
plt.xlabel('Classes')
plt.ylabel('Quantidade de produtos')
plt.ylim(0, 500)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

#para colocar o rotulo
for i, valor in enumerate(analise.groupby('Classificação_ABC').size()):
    plt.text(i, valor + 1, f'{valor:}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

#grafico de porcetagem de itens por classe
analise.groupby('Classificação_ABC')['Qtde_m2_%'].sum().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

plt.title('Distribuição de Itens por Classe ABC')
plt.xlabel('Classe')
plt.ylabel('Porcentagem (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

for i, valor in enumerate(analise.groupby('Classificação_ABC')['Qtde_m2_%'].sum()):
    plt.text(i, valor + 1, f'{valor:}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

#%% Separar os 72% dos itens para simplificar a análise
#principais itens que corresponde a 72% da movimentação
colunas = ['INC 03MM', 'INC 04MM', 'LAM INC 08MM', 'INC 06MM', 'LAM INC 06MM', 'LAM INC 10MM', 'INC 08MM', 'FUME 03MM','LAM REF PRATA 3012 08MM']

analise_principais = df_consumo_mensal[colunas]

#%% criar um visual para analisar a serie
df_dash = analise_principais.reset_index()

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Analise de estoque',style={'color': 'white', 'fontSize': '24px'},),
    dcc.Graph(id="serie-de-consumo"),
    html.P("Selecione produto:",style={'color': 'white','fontSize': '24px'}),
    dcc.Dropdown(
        id="vidros",
        options=[{'label': col, 'value': col} for col in df_dash.columns if col != 'Data'],
        value='INC 03MM',
        clearable=False,
    ),
])

@app.callback(
    Output("serie-de-consumo", "figure"),
    Input("vidros", "value"))
def display_time_series(vidros):
    fig = px.line(df_dash, x='Data', y=df_dash[vidros], title=f'Consumo de {vidros}')
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8050)#http://127.0.0.1:8050/

#%% Fazer a decomposição da serie
#Função de decomposição
#Livro de serie temporais com Python
def tspdecompose(df, modelo, periodo,title = None):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(df, model = modelo, period=periodo)
    fig = decomp.plot()
    fig.set_size_inches((9,7))

    if title is not None:
        plt.suptitle(title, y = 1.05)

    plt.tight_layout()
    plt.show()

#item inc 4mm
df_consumo_mensal_inc_04mm = analise_principais['INC 04MM']
tspdecompose(df_consumo_mensal_inc_04mm, 'additive',12)


#item lam inc 8mm
df_consumo_mensal_lam_inc_08mm = analise_principais['LAM INC 08MM']
tspdecompose(df_consumo_mensal_lam_inc_08mm,'additive',12)

#item lam inc 6mm
df_consumo_mensal_lam_inc_06mm = analise_principais['LAM INC 06MM']
tspdecompose(df_consumo_mensal_lam_inc_06mm,'additive',12)

#item  inc 6mm
df_consumo_mensal_inc_06mm = analise_principais['INC 06MM']
tspdecompose(df_consumo_mensal_inc_06mm,'additive',12)

#item lam inc 10mm
df_consumo_mensal_lam_inc_10mm = analise_principais['LAM INC 10MM']
tspdecompose(df_consumo_mensal_lam_inc_10mm, 'additive', 12)

#item inc 3mm
df_consumo_mensal_inc_03mm = analise_principais['INC 03MM']
tspdecompose(df_consumo_mensal_inc_03mm, 'additive', 12)

#%% Estudar os residuos

#estudar os resuiduos 4mm
resid_4mm = seasonal_decompose(df_consumo_mensal_inc_04mm, model='additive', period=12).resid.dropna()
plot_acf(resid_4mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos 4mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#estudar os resuiduos 8mm
resid_8mm = seasonal_decompose(df_consumo_mensal_lam_inc_08mm, model='additive', period=12).resid.dropna()
plot_acf(resid_8mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos 8mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#estudar os resuiduos lam 6mm
resid_lam_6mm =  seasonal_decompose(df_consumo_mensal_lam_inc_06mm, model='additive', period=12).resid.dropna()
plot_acf(resid_lam_6mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos lam 6mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#estudar os resuiduos 6mm
resid_6mm =  seasonal_decompose(df_consumo_mensal_inc_06mm, model='additive', period=12).resid.dropna()
plot_acf(resid_6mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos 6mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#estudar os resuiduos lam 10mm
resid_lam_10mm =  seasonal_decompose(df_consumo_mensal_lam_inc_10mm, model='additive', period=12).resid.dropna()
plot_acf(resid_lam_10mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos lam 10mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#estudar os resuiduos 3mm
resid_3mm =  seasonal_decompose(df_consumo_mensal_inc_03mm, model='additive', period=12).resid.dropna()
plot_acf(resid_3mm, lags=40)  # lags define quantos defasagens (lags) analisar
plt.title("Autocorrelação dos Resíduos inc 3mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação")
plt.show()

#%% Testar também o PACF (função de autocorrelação parcial) para detectar dependências diretas.

plot_pacf(resid_4mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos 4mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()

plot_pacf(resid_8mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos lam 8mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()

plot_pacf(resid_lam_6mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos lam 6mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()

plot_pacf(resid_6mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos inc 6mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()

plot_pacf(resid_lam_10mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos lam 10mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()

plot_pacf(resid_3mm, lags=35, method='ywm')  # 'ywm' = método de Yule-Walker modificado
plt.title("Autocorrelação Parcial dos Resíduos inc 3mm")
plt.xlabel("Defasagem (lag)")
plt.ylabel("Correlação parcial")
plt.show()



#%% Teste de Dickey-Fuller

#inc 4mm
resultado_4mm = adfuller(df_consumo_mensal_inc_04mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_4mm[0])
print('p-value:', resultado_4mm[1])
print('Nº de lags usados:', resultado_4mm[2])
print('Nº de observações usadas:', resultado_4mm[3])
print('Valores críticos:')
for chave, valor in resultado_4mm[4].items():
    print(f'   {chave}: {valor}')
    
#item lam inc 8mm
resultado_lam_8mm = adfuller(df_consumo_mensal_lam_inc_08mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_lam_8mm[0])
print('p-value:', resultado_lam_8mm[1])
print('Nº de lags usados:', resultado_lam_8mm[2])
print('Nº de observações usadas:', resultado_lam_8mm[3])
print('Valores críticos:')
for chave, valor in resultado_lam_8mm[4].items():
    print(f'   {chave}: {valor}')
    
#item lam inc 6mm
resultado_lam_6mm = adfuller(df_consumo_mensal_lam_inc_06mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_lam_6mm[0])
print('p-value:', resultado_lam_6mm[1])
print('Nº de lags usados:', resultado_lam_6mm[2])
print('Nº de observações usadas:', resultado_lam_6mm[3])
print('Valores críticos:')
for chave, valor in resultado_lam_6mm[4].items():
    print(f'   {chave}: {valor}')

#item inc 6mm
resultado_6mm = adfuller(df_consumo_mensal_inc_06mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_6mm[0])
print('p-value:', resultado_6mm[1])
print('Nº de lags usados:', resultado_6mm[2])
print('Nº de observações usadas:', resultado_6mm[3])
print('Valores críticos:')
for chave, valor in resultado_6mm[4].items():
    print(f'   {chave}: {valor}')

#item lam inc 10mm
resultado_lam_10mm = adfuller(df_consumo_mensal_lam_inc_10mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_lam_10mm[0])
print('p-value:', resultado_lam_10mm[1])
print('Nº de lags usados:', resultado_lam_10mm[2])
print('Nº de observações usadas:', resultado_lam_10mm[3])
print('Valores críticos:')
for chave, valor in resultado_lam_10mm[4].items():
    print(f'   {chave}: {valor}')

#item inc 3mm
resultado_3mm = adfuller(df_consumo_mensal_inc_03mm.dropna())

# Resultados principais:
print('ADF Statistic:', resultado_3mm[0])
print('p-value:', resultado_3mm[1])
print('Nº de lags usados:', resultado_3mm[2])
print('Nº de observações usadas:', resultado_3mm[3])
print('Valores críticos:')
for chave, valor in resultado_3mm[4].items():
    print(f'   {chave}: {valor}')

#%% Analise descritiva 

#descritivo do Inc 4mm
df_consumo_mensal_inc_04mm.describe()

#descritivo do lam inc 8mm
df_consumo_mensal_lam_inc_08mm.describe()

#%% Separar dados de treino e teste

#dados de teste e treino 4mm
df_consumo_mensal_inc_04mm_treino = df_consumo_mensal_inc_04mm.iloc[:-12]
df_consumo_mensal_inc_04mm_teste = df_consumo_mensal_inc_04mm.iloc[-12:]

#dados de teste e treino lam 8mm
df_consumo_mensal_lam_inc_08mm_treino = df_consumo_mensal_lam_inc_08mm.iloc[:-12]
df_consumo_mensal_lam_inc_08mm_teste = df_consumo_mensal_lam_inc_08mm.iloc[-12:]

#%% Separar dados de treino 2 e teste 2

#dados de teste e treino 4mm
df_consumo_mensal_inc_04mm_treino_v2 = df_consumo_mensal_inc_04mm.iloc[:-6]
df_consumo_mensal_inc_04mm_teste_v2 = df_consumo_mensal_inc_04mm.iloc[-6:]

#dados de teste e treino lam 8mm
df_consumo_mensal_lam_inc_08mm_treino_v2 = df_consumo_mensal_lam_inc_08mm.iloc[:-6]
df_consumo_mensal_lam_inc_08mm_teste_v2 = df_consumo_mensal_lam_inc_08mm.iloc[-6:]
#%% Modelo Prophet 4mm

#criar df para datas treino
df_prophet_treino = pd.DataFrame()
df_prophet_treino['ds'] = df_consumo_mensal_inc_04mm_treino.index
df_prophet_treino['y'] = df_consumo_mensal_inc_04mm_treino.values

#criar df para testes
df_prophet_teste = pd.DataFrame()
df_prophet_teste['ds'] = df_consumo_mensal_inc_04mm_teste.index
df_prophet_teste['y'] = df_consumo_mensal_inc_04mm_teste.values

#criar modelo
modelo_4mm = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
modelo_4mm.fit(df_prophet_treino)

#criando um df de previsão
df_futuro_inc4mm_treino = modelo_4mm.make_future_dataframe(periods=12, freq='ME')
df_previsao_inc4mm_treino = modelo_4mm.predict(df_futuro_inc4mm_treino)

#gráfico da previsão
modelo_4mm.plot(df_previsao_inc4mm_treino, figsize=(10, 5))
plt.title('Previsão Inc 04mm',loc='left', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

#comparar com o teste
fig1 =modelo_4mm.plot(df_previsao_inc4mm_treino, figsize=(10, 5))
plt.plot(df_prophet_teste['ds'],df_prophet_teste['y'], color='red')

#adicionar pontos de tendência detectados
a = add_changepoints_to_plot(fig1.gca(), modelo_4mm, df_previsao_inc4mm_treino)

plt.title('Previsão Inc 04mm com pontos de tendência', loc='left', fontsize=11)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

#%%Modelo Prophet 4mm v2

#criar df para datas treino
df_prophet_treino_v2 = pd.DataFrame()
df_prophet_treino_v2['ds'] = df_consumo_mensal_inc_04mm_treino.index
df_prophet_treino_v2['y'] = df_consumo_mensal_inc_04mm_treino.values

#criar df para testes
df_prophet_teste_v2 = pd.DataFrame()
df_prophet_teste_v2['ds'] = df_consumo_mensal_inc_04mm_teste.index
df_prophet_teste_v2['y'] = df_consumo_mensal_inc_04mm_teste.values

#criar modelo
modelo_4mm_v2 = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1, seasonality_mode='multiplicative')
modelo_4mm_v2.fit(df_prophet_treino_v2)

#criando um df de previsão
df_futuro_inc4mm_treino_v2 = modelo_4mm_v2.make_future_dataframe(periods=12, freq='ME')
df_previsao_inc4mm_treino_v2 = modelo_4mm_v2.predict(df_futuro_inc4mm_treino_v2)

#gráfico da previsão
modelo_4mm_v2.plot(df_previsao_inc4mm_treino_v2, figsize=(10, 5))
plt.title('Previsão Inc 04mm v2',loc='left', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

fig2 =modelo_4mm_v2.plot(df_previsao_inc4mm_treino_v2, figsize=(10, 5))
plt.plot(df_prophet_teste_v2['ds'],df_prophet_teste_v2['y'], color='red')

#adicionar pontos de tendência detectados
a2 = add_changepoints_to_plot(fig2.gca(), modelo_4mm_v2, df_previsao_inc4mm_treino)

plt.title('Previsão Inc 04mm com pontos de tendência', loc='left', fontsize=11)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

#%% modelo 4mm v3

#criar df para datas treino
df_prophet_treino_v3 = pd.DataFrame()
df_prophet_treino_v3['ds'] = df_consumo_mensal_inc_04mm_treino_v2.index
df_prophet_treino_v3['y'] = df_consumo_mensal_inc_04mm_treino_v2.values

#criar df para testes
df_prophet_teste_v3 = pd.DataFrame()
df_prophet_teste_v3['ds'] = df_consumo_mensal_inc_04mm_teste_v2.index
df_prophet_teste_v3['y'] = df_consumo_mensal_inc_04mm_teste_v2.values

#criar modelo
modelo_4mm_v3 = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1,seasonality_mode='multiplicative')
modelo_4mm_v3.fit(df_prophet_treino_v3)

#criando um df de previsão
df_futuro_inc4mm_treino_v3 = modelo_4mm_v3.make_future_dataframe(periods=6, freq='ME')
df_previsao_inc4mm_treino_v3 = modelo_4mm_v3.predict(df_futuro_inc4mm_treino_v3)

#gráfico da previsão
modelo_4mm_v3.plot(df_previsao_inc4mm_treino_v2, figsize=(10, 5))
plt.title('Previsão Inc 04mm v3',loc='left', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

fig3 =modelo_4mm_v3.plot(df_previsao_inc4mm_treino_v3, figsize=(10, 5))
plt.plot(df_prophet_teste_v3['ds'],df_prophet_teste_v3['y'], color='red')

#adicionar pontos de tendência detectados
a3 = add_changepoints_to_plot(fig3.gca(), modelo_4mm_v3, df_previsao_inc4mm_treino_v2)

plt.title('Previsão Inc 04mm com pontos de tendência', loc='left', fontsize=11)
plt.xlabel('Data', fontsize=11)
plt.ylabel('Quantidade', fontsize=11)
plt.show()

#%% Modelo Prophet lam 8mm
#criar df treino 
df_prophet_lam_inc_08mm_treino = pd.DataFrame()
df_prophet_lam_inc_08mm_treino['ds'] = df_consumo_mensal_lam_inc_08mm_treino.index
df_prophet_lam_inc_08mm_treino['y'] = df_consumo_mensal_lam_inc_08mm_treino.values

#criar df teste 
df_prophet_lam_inc_08mm_teste = pd.DataFrame()
df_prophet_lam_inc_08mm_teste['ds'] = df_consumo_mensal_lam_inc_08mm_teste.index
df_prophet_lam_inc_08mm_teste['y'] = df_consumo_mensal_lam_inc_08mm_teste.values

#criar modelo
modelo_prophet_lam_inc_08mm = Prophet(yearly_seasonality=True)
modelo_prophet_lam_inc_08mm.fit(df_prophet_lam_inc_08mm_treino)

#criar as previsões
df_futuro_lam_inc_08mm_treino = modelo_prophet_lam_inc_08mm.make_future_dataframe(periods=12, freq='ME')
df_previsao_lam_inc_08mm_treino = modelo_prophet_lam_inc_08mm.predict(df_futuro_lam_inc_08mm_treino)

#%% Modelo Arima 4mm
#usar o autoarima
modelo_arima_4mm = auto_arima(df_consumo_mensal_inc_04mm_treino,seasonal=True, m=12,trace =True)

#ajustar o modelo
mod_4mm = ARIMA(df_consumo_mensal_inc_04mm_treino, order=(2, 0, 0), seasonal_order=(0, 0, 1, 12)).fit()

print(mod_4mm.summary())

# Número de passos = tamanho do conjunto de teste
n_periodos = len(df_consumo_mensal_inc_04mm_teste)

# Fazendo a previsão do modelo auto_arima
previsao_arima_4mm = modelo_arima_4mm.predict(n_periods=n_periodos)

#%% Modelo Arima 4mm
#%% Com amostra reduzida
modelo_arima_4mm_v2 = auto_arima(df_consumo_mensal_inc_04mm_treino_v2, seasonal=True, m=12, trace=True)

#ajustar o modelo Arima(0,0,2)(0,0,1)
mod_4mm_v2 = ARIMA(df_consumo_mensal_inc_04mm_treino_v2, order=(0, 0, 2), seasonal_order=(0, 0, 1, 12)).fit()

print(mod_4mm_v2.summary())

# Número de passos = tamanho do conjunto de teste
n_periodos_v2 = len(df_consumo_mensal_inc_04mm_teste_v2)

# Fazendo a previsão do modelo auto_arima
previsao_arima_4mm_v2 = modelo_arima_4mm_v2.predict(n_periods=n_periodos_v2)
#%% Modelo Arima 8mm

#usar o autoarima
modelo_arima_lam_8mm = auto_arima(df_consumo_mensal_lam_inc_08mm_treino,seasonal=True, m=12,trace =True)

print(modelo_arima_lam_8mm.summary())

# Fazendo a previsão do modelo auto_arima
previsao_arima_lam_8mm = modelo_arima_lam_8mm.predict(n_periods=12)

#%% Modelo Sarimax 4mm

# Ajustar o modelo
modelo_sarimax_4mm = SARIMAX(df_consumo_mensal_inc_04mm_treino,
                       order=(2,0,0),
                       seasonal_order=(0,0,1,12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
resultado_sarimax_4mm = modelo_sarimax_4mm.fit()

# Previsão
forecast_auto_4mm = resultado_sarimax_4mm.forecast(steps=len(df_consumo_mensal_inc_04mm_teste), disp=False)

print(forecast_auto_4mm)

#%% Modelo Sarimax lam 8mm

# Ajustar o modelo
modelo_sarimax_lam_8mm = SARIMAX(df_consumo_mensal_lam_inc_08mm_treino,
                       order=(2,1,1),
                       seasonal_order=(2,0,0,12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
resultado_sarimax_lam_8mm = modelo_sarimax_lam_8mm.fit()

# Previsão
forecast_auto_lam_8mm = resultado_sarimax_lam_8mm.forecast(steps=12, disp=False)

print(forecast_auto_lam_8mm)

#%% Modelo Holt-Winters 4mm


modelo_hw_4mm = ExponentialSmoothing(df_consumo_mensal_inc_04mm_treino, trend='add', seasonal='add', seasonal_periods=12).fit()
print(modelo_hw_4mm.summary())

# Forecast com Holt-Winters
previsoes_hw_4mm = modelo_hw_4mm.forecast(steps=12)

#%% Modelo Holt-Winters lam 8mm

modelo_hw_lam_8mm = ExponentialSmoothing(df_consumo_mensal_lam_inc_08mm_treino, trend='add', seasonal='add', seasonal_periods=12).fit()
print(modelo_hw_lam_8mm.summary())

# Forecast com Holt-Winters
previsoes_hw_lam_8mm = modelo_hw_lam_8mm.forecast(steps=12)

#%% Modelo XGBoot 4mm

df_xg_4mm = df_consumo_mensal_inc_04mm.copy().to_frame()
df_xg_4mm = df_xg_4mm.reset_index(drop=True)
df_xg_4mm = df_xg_4mm.rename(columns={df_xg_4mm.columns[0]: 'INC_04MM'})

# Função para criar features de lags
def criar_lags(data, col, n_lags):
    for lag in range(1, n_lags+1):
        data[f'lag_{lag}'] = data[col].shift(lag)
    return data

# Criar lags
n_lags = 12  # 1 ano de lags mensais
df_xg_4mm = criar_lags(df_xg_4mm, col='INC_04MM', n_lags=n_lags)

# Remover valores nulos gerados pelos lags
df_xg_4mm = df_xg_4mm.dropna().reset_index(drop=True)

# Definir X (features) e y (target)
X = df_xg_4mm.drop(columns=['INC_04MM'])
y = df_xg_4mm['INC_04MM']

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Criar e treinar o XGBoost
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

#%% Modelo XGBoot lam 8mm

df_xg_lam_8mm = df_consumo_mensal_lam_inc_08mm.copy().to_frame()
df_xg_lam_8mm = df_xg_lam_8mm.reset_index(drop=True)
df_xg_lam_8mm = df_xg_lam_8mm.rename(columns={df_xg_lam_8mm.columns[0]: 'LAM_INC_08MM'})

# Criar lags
n_lags = 12  # 1 ano de lags mensais
df_xg_lam_8mm = criar_lags(df_xg_lam_8mm, col='LAM_INC_08MM', n_lags=n_lags)

# Remover valores nulos gerados pelos lags
df_xg_lam_8mm = df_xg_lam_8mm.dropna().reset_index(drop=True)

# Definir X (features) e y (target)
X_8mm = df_xg_lam_8mm.drop(columns=['LAM_INC_08MM'])
y_8mm = df_xg_lam_8mm['LAM_INC_08MM']

# Separar treino e teste
X_train_8mm, X_test_8mm, y_train_8mm, y_test_8mm = train_test_split(X_8mm, y_8mm, shuffle=False, test_size=0.2)

# Criar e treinar o XGBoost
model_lam_8mm = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_lam_8mm.fit(X_train_8mm, y_train_8mm)

# Fazer previsões
y_pred_lam_8mm = model.predict(X_test_8mm)

#%%verificar eficiencia dos modelos 

#Prophet
#4mm
df_4mm = df_previsao_inc4mm_treino[['ds','yhat']]
df_comparacao = pd.merge(df_4mm, df_prophet_teste, on='ds', how='inner')
mse = mean_squared_error(df_comparacao['y'],df_comparacao['yhat'])
rmse = np.sqrt(mse)
erro_relativo = (rmse / df_comparacao['y'].mean()) * 100
print(f'MSE: {mse}, RMSE: {rmse}, RMSE Relativo: {erro_relativo:.2f}%')

#Prophet v2
#4mm
df_4mm_v2 = df_previsao_inc4mm_treino[['ds','yhat']]
df_comparacao_v2 = pd.merge(df_4mm_v2, df_prophet_teste_v2, on='ds', how='inner')
mse_v2 = mean_squared_error(df_comparacao_v2['y'],df_comparacao_v2['yhat'])
rmse_v2 = np.sqrt(mse_v2)
erro_relativo_v2 = (rmse_v2 / df_comparacao_v2['y'].mean()) * 100
print(f'MSE: {mse_v2}, RMSE: {rmse_v2}, RMSE Relativo: {erro_relativo_v2:.2f}%')

#Prophet v3
#4mm
df_4mm_v3 = df_previsao_inc4mm_treino_v2[['ds','yhat']]
df_comparacao_v3 = pd.merge(df_4mm_v3, df_prophet_teste_v3, on='ds', how='inner')
mse_v3 = mean_squared_error(df_comparacao_v3['y'],df_comparacao_v3['yhat'])
rmse_v3 = np.sqrt(mse_v3)
erro_relativo_v3 = (rmse_v3 / df_comparacao_v3['y'].mean()) * 100
print(f'MSE: {mse_v3}, RMSE: {rmse_v3}, RMSE Relativo: {erro_relativo_v3:.2f}%')

#Lam 8mm
df_lam_inc_8mm = df_previsao_lam_inc_08mm_treino[['ds','yhat']]
df_comparacao_lam_inc_8mm = pd.merge(df_lam_inc_8mm, df_prophet_lam_inc_08mm_teste, on='ds', how='inner')
mse_lam_8mm = mean_squared_error(df_comparacao_lam_inc_8mm['y'],df_comparacao_lam_inc_8mm['yhat'])
rmse_lam_8mm = np.sqrt(mse_lam_8mm)
erro_relativo_lam_8mm = (rmse_lam_8mm / df_comparacao_lam_inc_8mm['y'].mean()) * 100
print(f'MSE: {mse_lam_8mm}, RMSE: {rmse_lam_8mm}, RMSE Relativo: {erro_relativo_lam_8mm:.2f}%')

#Arima
#4mm
df_arima_4mm = pd.DataFrame()
df_arima_4mm['ds'] = previsao_arima_4mm.index
df_arima_4mm['yhat'] = previsao_arima_4mm[0]
df_comparacao_arima = pd.merge(df_arima_4mm, df_prophet_teste, on='ds', how='inner')
mse_arima_4mm = mean_squared_error(df_comparacao_arima['y'],df_comparacao_arima['yhat'])
rmse_arima_4mm = np.sqrt(mse_arima_4mm)
erro_relativo_arima_4mm = (rmse_arima_4mm / df_comparacao_arima['y'].mean()) * 100
print(f'MSE: {mse_arima_4mm}, RMSE: {rmse_arima_4mm}, RMSE Relativo: {erro_relativo_arima_4mm:.2f}%')

#Arima v2
#4mm
df_arima_4mm_v2 = pd.DataFrame()
df_arima_4mm_v2['ds'] = previsao_arima_4mm_v2.index
df_arima_4mm_v2['yhat'] = previsao_arima_4mm_v2[0]
df_comparacao_arima_v2 = pd.merge(df_arima_4mm_v2, df_prophet_teste_v2, on='ds', how='inner')
mse_arima_4mm_v2 = mean_squared_error(df_comparacao_arima_v2['y'],df_comparacao_arima_v2['yhat'])
rmse_arima_4mm_v2 = np.sqrt(mse_arima_4mm_v2)
erro_relativo_arima_4mm_v2 = (rmse_arima_4mm_v2 / df_comparacao_arima_v2['y'].mean()) * 100
print(f'MSE: {mse_arima_4mm_v2}, RMSE: {rmse_arima_4mm_v2}, RMSE Relativo: {erro_relativo_arima_4mm_v2:.2f}%')

#lam 8mm
df_arima_lam_8mm = pd.DataFrame()
df_arima_lam_8mm['ds'] = previsao_arima_lam_8mm.index
df_arima_lam_8mm['yhat'] = previsao_arima_lam_8mm[0]
df_comparacao_arima_lam_8mm = pd.merge(df_arima_lam_8mm,df_prophet_lam_inc_08mm_teste,on='ds',how='inner')
mse_arima_lam_8mm = mean_squared_error(df_comparacao_arima_lam_8mm['y'],df_comparacao_arima_lam_8mm['yhat'])
rmse_arima_lam_8mm = np.sqrt(mse_arima_lam_8mm)
erro_relativo_arima_lam_8mmmm = (rmse_arima_lam_8mm / df_comparacao_arima_lam_8mm['y'].mean()) * 100
print(f'MSE: {mse_arima_lam_8mm}, RMSE: {rmse_arima_lam_8mm}, RMSE Relativo: {erro_relativo_arima_lam_8mmmm:.2f}%')

#Sarimax
#4mm
df_sarimax_4mm = pd.DataFrame()
df_sarimax_4mm['ds'] = forecast_auto_4mm.index
df_sarimax_4mm['yhat'] = forecast_auto_4mm[0]
df_comparacao_sarimax_4mm = pd.merge(df_sarimax_4mm,df_prophet_teste, on='ds',how='inner')
mse_sarimax_4mm = mean_squared_error(df_comparacao_sarimax_4mm['y'],df_comparacao_sarimax_4mm['yhat'])
rmse_sarimax_4mm = np.sqrt(mse_sarimax_4mm)
erro_relativo_sarimax_4mm = (rmse_sarimax_4mm / df_comparacao_sarimax_4mm['y'].mean()) * 100
print(f'MSE: {mse_sarimax_4mm}, RMSE: {rmse_sarimax_4mm}, RMSE Relativo: {erro_relativo_sarimax_4mm:.2f}%')

#8mm
df_sarimax_lam_8mm = pd.DataFrame()
df_sarimax_lam_8mm['ds'] = forecast_auto_lam_8mm.index
df_sarimax_lam_8mm['yhat'] = forecast_auto_lam_8mm[0]
df_comparacao_sarimax_lam_8mm = pd.merge(df_sarimax_lam_8mm,df_prophet_lam_inc_08mm_teste, on='ds',how='inner')
mse_sarimax_lam_8mm = mean_squared_error(df_comparacao_sarimax_lam_8mm['y'],df_comparacao_sarimax_lam_8mm['yhat'])
rmse_sarimax_lam_8mm = np.sqrt(mse_sarimax_lam_8mm)
erro_relativo_sarimax_lam_8mm = (rmse_sarimax_lam_8mm / df_comparacao_sarimax_lam_8mm['y'].mean()) * 100
print(f'MSE: {mse_sarimax_lam_8mm}, RMSE: {rmse_sarimax_lam_8mm}, RMSE Relativo: {erro_relativo_sarimax_lam_8mm:.2f}%')

#Hold-winters
#4mm
df_hw_4mm = pd.DataFrame()
df_hw_4mm['ds'] =previsoes_hw_4mm.index
df_hw_4mm['yhat'] = previsoes_hw_4mm[0]
df_comparacao_hw_4mm = pd.merge(df_hw_4mm,df_prophet_teste,on='ds',how='inner')
mse_hw_4mm = mean_squared_error(df_comparacao_hw_4mm['y'],df_comparacao_hw_4mm['yhat'])
rmse_hw_4mm = np.sqrt(mse_hw_4mm)
erro_relativo_hw_4mm =(rmse_hw_4mm / df_comparacao_hw_4mm['y'].mean())*100 
print(f'MSE: {mse_hw_4mm}, RMSE: {rmse_hw_4mm}, RMSE Relativo: {erro_relativo_hw_4mm:.2f}%')

#lam 8mm
df_hw_lam_8mm = pd.DataFrame()
df_hw_lam_8mm['ds'] =previsoes_hw_lam_8mm.index
df_hw_lam_8mm['yhat'] = previsoes_hw_lam_8mm[0]
df_comparacao_hw_lam_8mm = pd.merge(df_hw_lam_8mm,df_prophet_lam_inc_08mm_teste,on='ds',how='inner')
mse_hw_lam_8mm = mean_squared_error(df_comparacao_hw_lam_8mm['y'],df_comparacao_hw_lam_8mm['yhat'])
rmse_hw_lam_8mm = np.sqrt(mse_hw_lam_8mm)
erro_relativo_hw_lam_8mm =(rmse_hw_lam_8mm / df_comparacao_hw_lam_8mm['y'].mean())*100 
print(f'MSE: {mse_hw_lam_8mm}, RMSE: {rmse_hw_lam_8mm}, RMSE Relativo: {erro_relativo_hw_lam_8mm:.2f}%')

#XGBoost
#4mm
mse_xgb = mean_squared_error(y_test, y_pred)
rmse_xgb = np.sqrt(mse_xgb)
erro_relativo_xgb = (rmse_xgb / y_test.mean()) * 100
print(f'MSE: {mse_xgb:.2f},RMSE: {rmse_xgb}, RMSE Relativo: {erro_relativo_xgb:.2f}%')


#lam 8mm
mse_xgb_8mm = mean_squared_error(y_test_8mm, y_pred_lam_8mm)
rmse_xgb_8mm = np.sqrt(mse_xgb_8mm)
erro_relativo_xgb_8mm = (rmse_xgb_8mm / y_test_8mm.mean()) * 100
print(f'MSE: {mse_xgb_8mm:.2f},RMSE: {rmse_xgb_8mm}, RMSE Relativo: {erro_relativo_xgb_8mm:.2f}%')


#%%Resultados
# Lista para armazenar resultados
resultados = []

resultados.append({
    "Modelo": "Prophet",
    "Série": "04mm",
    "MSE": mse,
    "RMSE": rmse,
    "RMSE Relativo (%)": erro_relativo
    })

resultados.append({
    "Modelo": "Prophet",
    "Série": "Lam 08mm",
    "MSE": mse_lam_8mm,
    "RMSE": rmse_lam_8mm,
    "RMSE Relativo (%)": erro_relativo_lam_8mm
    })

resultados.append({
    "Modelo": "ARIMA",
    "Série": "04mm",
    "MSE": mse_arima_4mm,
    "RMSE": rmse_arima_4mm,
    "RMSE Relativo (%)": erro_relativo_arima_4mm
    })

resultados.append({
    "Modelo": "ARIMA",
    "Série": "Lam 08mm",
    "MSE": mse_arima_lam_8mm,
    "RMSE": rmse_arima_lam_8mm,
    "RMSE Relativo (%)": erro_relativo_arima_lam_8mmmm
    })

resultados.append({
    "Modelo": "ARIMA_v2",
    "Série": "04mm",
    "MSE": mse_arima_4mm_v2,
    "RMSE":  rmse_arima_4mm_v2,
    "RMSE Relativo (%)": erro_relativo_arima_4mm_v2
    })

resultados.append({
    "Modelo": "Sarimax",
    "Série": "04mm",
    "MSE": mse_sarimax_4mm,
    "RMSE": rmse_sarimax_4mm,
    "RMSE Relativo (%)": erro_relativo_sarimax_4mm
    })

resultados.append({
    "Modelo": "Sarimax",
    "Série": "Lam 08mm",
    "MSE": mse_sarimax_lam_8mm,
    "RMSE": rmse_sarimax_lam_8mm,
    "RMSE Relativo (%)": erro_relativo_sarimax_lam_8mm
    })

resultados.append({
    "Modelo": "Holt-Winters",
    "Série": "04mm",
    "MSE": mse_hw_4mm,
    "RMSE": rmse_hw_4mm,
    "RMSE Relativo (%)": erro_relativo_hw_4mm
    })

resultados.append({
    "Modelo": "Holt-Winters",
    "Série": "Lam 08mm",
    "MSE": mse_hw_lam_8mm,
    "RMSE": rmse_hw_lam_8mm,
    "RMSE Relativo (%)": erro_relativo_hw_lam_8mm
    })

resultados.append({
    "Modelo": "XGBoost",
    "Série": "04mm",
    "MSE": mse_xgb,
    "RMSE": rmse_xgb,
    "RMSE Relativo (%)": erro_relativo_xgb
    })

resultados.append({
    "Modelo": "XGBoost",
    "Série": "Lam 08mm",
    "MSE": mse_xgb_8mm,
    "RMSE": rmse_xgb_8mm,
    "RMSE Relativo (%)": erro_relativo_xgb_8mm
    })

resultados.append({
    "Modelo": "Prophet_v2",
    "Série": "04mm",
    "MSE": mse,
    "RMSE": rmse,
    "RMSE Relativo (%)": erro_relativo
    })

# Criar dataframe final
df_resultados = pd.DataFrame(resultados)

print(df_resultados)

#passar para o excel
df_resultados.to_excel("resultados.xlsx")



