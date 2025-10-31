
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy import stats


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard de Saúde"


url_tema_claro = dbc.themes.PULSE
url_tema_escuro = dbc.themes.CYBORG
tema_claro = 'pulse'
tema_escuro = 'cyborg'


df = pd.read_csv('Health_Risk_Dataset.csv')


df["Febre"] = (df["Temperature"] > 37.5).astype(int)


df["Risk_Level"] = df["Risk_Level"].map({
    "Normal": "Normal",
    "Low": "Baixo",
    "Medium": "Médio",
    "High": "Alto"
})
df["Consciousness"] = df["Consciousness"].map({
    "A": "Alerta",
    "P": "Resposta à Dor",
    "C": "Confuso",
    "V": "Resposta Verbal",
    "U": "Inconsciente"
})




novos_nomes = {
    'Patient_ID': 'ID_Paciente',
    'Respiratory_Rate': 'Respiracoes_Minuto',  
    'Oxygen_Saturation': 'Saturacao_Oxigenio',
    'O2_Scale': 'Escala_O2',
    'Systolic_BP': 'Pressao_Sistolica',
    'Heart_Rate': 'Frequencia_Cardiaca',
    'Temperature': 'Temperatura',
    'Consciousness': 'Nivel_Consciencia',
    'On_Oxygen': 'Usa_Oxigenio',
    'Risk_Level': 'Nivel_Risco'
}
df.rename(columns=novos_nomes, inplace=True)

escala_o2_map = {
    0: "Sem oxigênio",
    1: "Oxigênio leve",
    2: "Oxigênio moderado",
    3: "Oxigênio intenso"
}


df['Escala_O2'] = df['Escala_O2'].map(escala_o2_map)


df['Escala_O2'] = df['Escala_O2'].astype(object)

categorical_cols = ["Nivel_Risco", "Nivel_Consciencia", "Usa_Oxigenio", "Febre","Escala_O2"]
numeric_cols = ["Saturacao_Oxigenio", "Respiracoes_Minuto", "Pressao_Sistolica",
                "Frequencia_Cardiaca", "Temperatura"]
for col in categorical_cols:
    df[col] = df[col].astype(str)

nivel_risco_options = [{'label': v, 'value': v} for v in sorted(df["Nivel_Risco"].dropna().unique())]
consciencia_options = [{'label': v, 'value': v} for v in sorted(df["Nivel_Consciencia"].dropna().unique())]
oxigenio_options = [{'label': v, 'value': v} for v in sorted(df["Usa_Oxigenio"].dropna().unique())]
febre_options = [{'label': v, 'value': v} for v in sorted(df["Febre"].dropna().unique())]
metric_options = [{'label': col, 'value': col} for col in numeric_cols]
default_metric = numeric_cols[0]


TAB_STYLE_LIGHT = {
    'padding': '10px',
    'backgroundColor': '#ffffff',
    'color': '#000000',
    'border': '1px solid #ccc'
}
TAB_STYLE_DARK = {
    'padding': '10px',
    'backgroundColor': '#111111',
    'color': '#ffffff',
    'border': '1px solid #444'
}
TAB_SELECTED_LIGHT = {
    'padding': '10px',
    'backgroundColor': '#e0e0e0',
    'color': '#000000',
    'border': '1px solid #ccc'
}
TAB_SELECTED_DARK = {
    'padding': '10px',
    'backgroundColor': '#000000',
    'color': '#ffffff',
    'border': '1px solid #444'
}


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Monitoramento de Saúde"), width=8),
        dbc.Col(ThemeSwitchAIO(aio_id='tema_switch', themes=[url_tema_claro, url_tema_escuro]), width=4)
    ], className="my-3"),

    dbc.Row([
        dbc.Col(
            dcc.Tabs(
                id="tabs_dashboard",
                value='tab_analise',
                children=[
                    dcc.Tab(label='Análise', value='tab_analise', style=TAB_STYLE_LIGHT, selected_style=TAB_SELECTED_LIGHT),
                    dcc.Tab(label='Predição', value='tab_predicao', style=TAB_STYLE_LIGHT, selected_style=TAB_SELECTED_LIGHT)
                ]
            )
        )
    ], className="mb-3"),

    html.Div(id='tabs_content')
], fluid=True)




@app.callback(
    Output('tabs_dashboard', 'children'),
    Input(ThemeSwitchAIO.ids.switch('tema_switch'), 'value'),
    State('tabs_dashboard', 'value')
)
def update_tabs_theme(is_light, active_tab):
    if is_light:
        style = TAB_STYLE_LIGHT
        selected = TAB_SELECTED_LIGHT
    else:
        style = TAB_STYLE_DARK
        selected = TAB_SELECTED_DARK

    tabs = [
        dcc.Tab(label='Análise', value='tab_analise', style=style, selected_style=selected),
        dcc.Tab(label='Predição', value='tab_predicao', style=style, selected_style=selected)
    ]
    return tabs


@app.callback(
    Output('tabs_content', 'children'),
    Input('tabs_dashboard', 'value')
)
def render_tabs(tab):
    if tab == 'tab_analise':
        return html.Div([

            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Descrição das Colunas do Dataset",
                        id="toggle_desc", color="secondary",
                        className="mb-3", n_clicks=0
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody([
                                html.P(f"{c}: {desc}") for c, desc in {
                                    "ID_Paciente": "Identificador único do paciente",
                                    "Respiracoes_Minuto": "Número de respirações por minuto",
                                    "Saturacao_Oxigenio": "Nível de saturação de oxigênio (%)",
                                    "Escala_O2": "Escala utilizada na Terapia de oxigênio",
                                    "Pressao_Sistolica": "Pressão arterial sistólica",
                                    "Frequencia_Cardiaca": "Batimentos por minuto",
                                    "Temperatura": "Temperatura corporal",
                                    "Nivel_Consciencia": "Nível de consciência",
                                    "Usa_Oxigenio": "Uso de oxigênio suplementar",
                                    "Nivel_Risco": "Risco de saúde",
                                    "Febre": "Paciente com febre"
                                }.items()
                            ])
                        ),
                        id="collapse_desc", is_open=False
                    )
                ], width=12)
            ]),

            # Filtros
            dbc.Row([
                dbc.Col([
                    html.H5("Nível de Risco"),
                    dcc.Dropdown(
                        id='nivel_risco_filter',
                        options=nivel_risco_options,
                        value=[v['value'] for v in nivel_risco_options],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.H5("Nível de Consciência"),
                    dcc.Dropdown(
                        id='consciencia_filter',
                        options=consciencia_options,
                        value=[v['value'] for v in consciencia_options],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.H5("Uso de Oxigênio"),
                    dcc.Dropdown(
                        id='oxigenio_filter',
                        options=oxigenio_options,
                        value=[v['value'] for v in oxigenio_options],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.H5("Febre"),
                    dcc.Dropdown(
                        id='febre_filter',
                        options=febre_options,
                        value=[v['value'] for v in febre_options],
                        multi=True
                    )
                ], width=3)
            ], className="my-3"),

            # Gráficos
            dbc.Row([
                dbc.Col([
                    html.H5("Métrica Numérica"),
                    dcc.Dropdown(
                        id='metric_escolha',
                        options=metric_options,
                        value=default_metric,
                        multi=False
                    )
                ], width=4),
                dbc.Col([
                    html.H5("Agrupar por"),
                    dcc.Dropdown(
                        id='groupby_escolha',
                        options=[{'label': col, 'value': col} for col in categorical_cols],
                        value='Nivel_Risco',
                        multi=False
                    )
                ], width=4)
            ], className="my-3"),

            dbc.Row([dbc.Col(dcc.Graph(id='bar_chart'), width=12)], className="my-3"),

            dbc.Row([
                dbc.Col([
                    html.H5("Comparação (Scatter)"),
                    dcc.Dropdown(
                        id='scatter_comparison',
                        options=[],
                        value=None,
                        multi=False
                    )
                ], width=4)
            ], className="my-3"),

            dbc.Row([dbc.Col(dcc.Graph(id='scatter_chart'), width=12)], className="my-3"),

            dbc.Row([
                dbc.Col([
                    html.H5("Categoria para Gráfico de Pizza"),
                    dcc.Dropdown(
                        id='pie_category',
                        options=[{'label': col, 'value': col} for col in categorical_cols],
                        value='Nivel_Risco'
                    ),
                    dcc.Graph(id='pie_chart', style={'height': '500px'})
                ], width=12)
            ], className="my-3"),
            dbc.Row([dbc.Col(dcc.Graph(id='correlation_heatmap'), width=12)], className="my-3")
        ])

    elif tab == 'tab_predicao':
        return html.Div([
            # Previsão simples
            dbc.Row([
                dbc.Col([
                    html.H5("Prever Temperatura"),
                    dbc.Input(id="input_fc", type="number", placeholder="Frequência Cardíaca"),
                    dbc.Input(id="input_ox", type="number", placeholder="Saturação de Oxigênio"),
                    dbc.Button("Prever", id="btn_prever", color="primary", className="mt-2"),
                    html.Div(id="saida_previsao", style={'marginTop': '10px'})
                ], width=4)
            ]),

            # Regressão linear manual
            dbc.Row([
                dbc.Col([
                    html.H5("Escolha as variáveis para Regressão Linear"),
                    dcc.Dropdown(
                        id='linear_input1',
                        options=[{'label': c, 'value': c} for c in numeric_cols],
                        placeholder="Entrada 1"
                    ),
                    dcc.Dropdown(
                        id='linear_input2',
                        options=[{'label': c, 'value': c} for c in numeric_cols],
                        placeholder="Entrada 2"
                    ),
                    dcc.Dropdown(
                        id='linear_output',
                        options=[{'label': c, 'value': c} for c in numeric_cols],
                        placeholder="Saída"
                    ),
                    dcc.Input(id='linear_val1', type='number', placeholder="Valor de X1"),
                    dcc.Input(id='linear_val2', type='number', placeholder="Valor de X2"),
                    html.Button("Rodar Regressão", id='run_linear_regression', n_clicks=0),
                    html.Div(id='linear_prediction_output', style={'marginTop': '10px'})
                ], width=6)
            ]),

            # Regressão linear automática
            dbc.Row([
                dbc.Col([
                    html.H5("Modelo de Regressão Linear Automático"),
                    dcc.Dropdown(
                        id='auto_regression_target',
                        options=[{'label': c, 'value': c} for c in numeric_cols],
                        placeholder="Selecione a variável alvo"
                    ),
                    html.Button("Rodar Modelo", id='run_auto_regression', n_clicks=0, className="mt-2"),
                    html.Div(id='auto_regression_output', style={'marginTop': '10px'})
                ], width=6)
            ]),

        # Intervalo de confiança
        dbc.Row([dbc.Col([
            html.H5("Intervalo de Confiança 95%"),
            dcc.Dropdown(
                id='ci_column',
                options=[{'label': c, 'value': c} for c in numeric_cols],
                placeholder="Selecione a coluna"
            ),
            html.Button("Calcular IC", id='btn_ci', n_clicks=0, className="mt-2"),
            html.Div(id='ci_output', style={'marginTop':'10px'})
        ], width=6)]),

        # p-value comparando com Frequência Cardíaca
        dbc.Row([dbc.Col([
            html.H5("p-value (comparação com Frequência Cardíaca)"),
            dcc.Dropdown(
                id='pvalue_column',
                options=[{'label': c, 'value': c} for c in numeric_cols if c != 'Frequencia_Cardiaca'],
                placeholder="Selecione a coluna"
            ),
            html.Button("Calcular p-value", id='btn_pvalue', n_clicks=0, className="mt-2"),
            html.Div(id='pvalue_output', style={'marginTop':'10px'})
        ], width=6)])
        ])


# Colapso descrição
@app.callback(
    Output("collapse_desc", "is_open"),
    Input("toggle_desc", "n_clicks"),
    [State("collapse_desc", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Gráfico de barras
@app.callback(
    Output('bar_chart', 'figure'),
    Input('nivel_risco_filter', 'value'),
    Input('consciencia_filter', 'value'),
    Input('oxigenio_filter', 'value'),
    Input('febre_filter', 'value'),
    Input('groupby_escolha', 'value'),
    Input('metric_escolha', 'value'),
    Input(ThemeSwitchAIO.ids.switch('tema_switch'), 'value')
)
def update_bar_chart(nivel_risco, consciencia, oxigenio, febre, groupby_col, metric, tema_switch):
    tema = tema_claro if tema_switch else tema_escuro
    df_filtered = df[
        df['Nivel_Risco'].isin(nivel_risco) &
        df['Nivel_Consciencia'].isin(consciencia) &
        df['Usa_Oxigenio'].isin(oxigenio) &
        df['Febre'].isin(febre)
    ]
    if groupby_col and metric:
        df_grouped = df_filtered.groupby(groupby_col)[metric].mean().reset_index()
        fig = px.bar(df_grouped, x=groupby_col, y=metric, color=groupby_col,
                     template=tema, title=f"Média de {metric} agrupada por {groupby_col}")
    else:
        fig = px.bar(template=tema, title="Selecione parâmetros")
    return fig

# Scatter opções
@app.callback(
    Output('scatter_comparison', 'options'),
    Input('metric_escolha', 'value')
)
def update_scatter_options(selected_metric):
    comparison_cols = [c for c in categorical_cols + numeric_cols if c != selected_metric]
    return [{'label': c, 'value': c} for c in comparison_cols]

# Scatter chart
@app.callback(
    Output('scatter_chart', 'figure'),
    Input('nivel_risco_filter', 'value'),
    Input('consciencia_filter', 'value'),
    Input('oxigenio_filter', 'value'),
    Input('febre_filter', 'value'),
    Input('metric_escolha', 'value'),
    Input('scatter_comparison', 'value'),
    Input(ThemeSwitchAIO.ids.switch('tema_switch'), 'value')
)
def update_scatter_chart(nivel_risco, consciencia, oxigenio, febre, metric, comparison, tema_switch):
    tema = tema_claro if tema_switch else tema_escuro
    df_filtered = df[
        df['Nivel_Risco'].isin(nivel_risco) &
        df['Nivel_Consciencia'].isin(consciencia) &
        df['Usa_Oxigenio'].isin(oxigenio) &
        df['Febre'].isin(febre)
    ]
    x_col = next((col for col in numeric_cols if col != metric), numeric_cols[0])
    color_col = comparison if comparison else 'Nivel_Risco'
    fig = px.scatter(df_filtered, x=x_col, y=metric, color=color_col,
                     hover_data=df_filtered.columns, template=tema, title=f"{metric} vs {x_col}")
    fig.update_layout(height=500)
    return fig

# Pie chart
@app.callback(
    Output('pie_chart', 'figure'),
    Input('nivel_risco_filter', 'value'),
    Input('consciencia_filter', 'value'),
    Input('oxigenio_filter', 'value'),
    Input('febre_filter', 'value'),
    Input('pie_category', 'value'),
    Input(ThemeSwitchAIO.ids.switch('tema_switch'), 'value') 
)
def update_pie(nivel_risco, consciencia, oxigenio, febre, category, tema_switch):
    df_filtered = df[
        df['Nivel_Risco'].isin(nivel_risco) &
        df['Nivel_Consciencia'].isin(consciencia) &
        df['Usa_Oxigenio'].isin(oxigenio) &
        df['Febre'].isin(febre)
    ]
    df_count = df_filtered[category].value_counts().reset_index()
    df_count.columns = [category, 'Count']
    tema = tema_claro if tema_switch else tema_escuro
    fig = px.pie(df_count, names=category, values='Count', title=f"Distribuição de {category}",
                 hole=0.3, template=tema)
    fig.update_layout(height=500)
    return fig

@app.callback(
    Output('correlation_heatmap', 'figure'),
    Input(ThemeSwitchAIO.ids.switch('tema_switch'), 'value')
)
def update_correlation_heatmap(is_light):
    tema = tema_claro if is_light else tema_escuro


    corr = df[numeric_cols].corr()

    # Cria heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Mapa de Correlação das Variáveis Numéricas',
        labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
        template=tema
    )
    fig.update_layout(height=600)
    return fig

# Previsão simples
@app.callback(
    Output("saida_previsao", "children"),
    Input("btn_prever", "n_clicks"),
    State("input_fc", "value"),
    State("input_ox", "value")
)
def prever_temperatura(n_clicks, fc, ox):
    if n_clicks is None or fc is None or ox is None:
        return "Preencha os campos e clique em Prever"
    X = df[['Frequencia_Cardiaca', 'Saturacao_Oxigenio']]
    y = df['Temperatura']
    modelo = LinearRegression()
    modelo.fit(X, y)
    temp_pred = modelo.predict([[fc, ox]])[0]
    return f"Temperatura prevista: {temp_pred:.2f} °C"

# Callbacks regressão linear
@app.callback(
    Output('linear_input1', 'options'),
    Input('linear_input2', 'value'),
    Input('linear_output', 'value')
)
def update_x1_options(x2_val, y_val):
    cols = [c for c in numeric_cols if c not in [x2_val, y_val]]
    return [{'label': c, 'value': c} for c in cols]

@app.callback(
    Output('linear_input2', 'options'),
    Input('linear_input1', 'value'),
    Input('linear_output', 'value')
)
def update_x2_options(x1_val, y_val):
    cols = [c for c in numeric_cols if c not in [x1_val, y_val]]
    return [{'label': c, 'value': c} for c in cols]

@app.callback(
    Output('linear_output', 'options'),
    Input('linear_input1', 'value'),
    Input('linear_input2', 'value')
)
def update_y_options(x1_val, x2_val):
    cols = [c for c in numeric_cols if c not in [x1_val, x2_val]]
    return [{'label': c, 'value': c} for c in cols]

@app.callback(
    Output('linear_prediction_output', 'children'),
    Input('run_linear_regression', 'n_clicks'),
    State('linear_input1', 'value'),
    State('linear_input2', 'value'),
    State('linear_output', 'value'),
    State('linear_val1', 'value'),
    State('linear_val2', 'value')
)


def run_linear_model(n_clicks, x1_col, x2_col, y_col, val1, val2):
    if not n_clicks:
        return "Aguardando execução..."
    if None in [x1_col, x2_col, y_col, val1, val2]:
        return "Preencha todos os campos!"
    
    X = df[[x1_col, x2_col]]
    y = df[y_col]
    
    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)
    y_imp = y.fillna(y.mean())
    
    modelo = LinearRegression()
    modelo.fit(X_imp, y_imp)
    
    pred = modelo.predict(imputer.transform([[val1, val2]]))[0]
    
    return html.Div([
        html.P(f"Variáveis: {x1_col}={val1}, {x2_col}={val2}"),
        html.P(f"Previsão de {y_col}: {pred:.2f}")
    ])

# Auto regressão linear

@app.callback(
    Output('auto_regression_output', 'children'),
    Input('run_auto_regression', 'n_clicks'),
    State('auto_regression_target', 'value')
)
def auto_linear_regression(n_clicks, target):
    if not n_clicks or not target:
        return "Aguardando execução..."
    
  
    features = [col for col in numeric_cols if col != target]
    X = df[features]
    y = df[target].fillna(df[target].mean())
    

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)
    
    
    modelo = LinearRegression()
    modelo.fit(X_imp, y)
    

    y_pred = modelo.predict(X_imp)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    
    coef_lines = [f"{col}: {coef:.3f}" for col, coef in zip(features, modelo.coef_)]
    
   
    lines = [f"Modelo para {target}", "", f"Intercepto: {modelo.intercept_:.3f}", "Coeficientes:"]
    lines.extend(coef_lines)
    lines.append(f"R²: {r2:.3f}")
    lines.append(f"RMSE: {rmse:.3f}")
    
    return html.Pre("\n".join(lines))

@app.callback(
    Output('ci_output', 'children'),
    Input('btn_ci', 'n_clicks'),
    State('ci_column', 'value')
)
def calcular_ic(n_clicks, coluna):
    if not n_clicks or not coluna:
        return "Aguardando execução..."
    
    dados = df[coluna].dropna()
    media = np.mean(dados)
    erro = stats.sem(dados) 
    intervalo = stats.t.interval(0.95, len(dados)-1, loc=media, scale=erro)
    
    return f"IC 95% para {coluna}: ({intervalo[0]:.3f}, {intervalo[1]:.3f})"

@app.callback(
    Output('pvalue_output', 'children'),
    Input('btn_pvalue', 'n_clicks'),
    State('pvalue_column', 'value')
)
def calcular_pvalue_com_fc(n_clicks, coluna):
    if not n_clicks or not coluna:
        return "Aguardando execução..."
    
    dados_coluna = df[coluna].dropna()
    dados_fc = df['Frequencia_Cardiaca'].dropna()
    
    min_len = min(len(dados_coluna), len(dados_fc))
    dados_coluna = dados_coluna.sample(min_len, random_state=42)
    dados_fc = dados_fc.sample(min_len, random_state=42)
    
    # Teste t de duas amostras (independente)
    stat, p = stats.ttest_ind(dados_coluna, dados_fc, equal_var=False)
    
    return f"Teste t de {coluna} vs Frequência Cardíaca: estatística={stat:.3f}, p-value={p:.3f}"

if __name__ == "__main__":
    app.run(debug=True)
