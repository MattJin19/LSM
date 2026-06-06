import base64
import io
import re

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# ---------------------------
# 辅助函数
# ---------------------------
def extract_dps(operation_str):
    """
    从类似 'Pre Op DAY 10' 或 'Post Op DAY 5' 的字符串中抽取 DPS
    'Pre' -> negative, 'Post' -> positive
    若匹配不上返回 NaN
    """
    try:
        if not isinstance(operation_str, str):
            return np.nan
        match = re.search(r'(Pre|Post)\s*Op\s*DAY\s*(\d+)', operation_str, re.IGNORECASE)
        if match:
            day = int(match.group(2))
            return -day if 'Pre' in match.group(1) else day
        return np.nan
    except Exception:
        return np.nan


def detect_peaks_and_valleys(y, distance=7, prominence=1.0):
    """
    在一维数组 y 上检测峰和谷（谷通过对 -y 检测峰实现）。
    返回: dict with peaks_idx, peaks_props, valleys_idx, valleys_props
    """
    peaks_idx, peaks_props = find_peaks(y, distance=distance, prominence=prominence)
    valleys_idx, valleys_props = find_peaks(-y, distance=distance, prominence=prominence)
    return {
        "peaks_idx": peaks_idx,
        "peaks_props": peaks_props,
        "valleys_idx": valleys_idx,
        "valleys_props": valleys_props
    }


def prepare_df(df, operation_col=None, dps_col=None, steps_col=None, smooth_col=None):
    """
    返回 DataFrame，确保含有 DPS 列和数值型 steps 列
    """
    df = df.copy()

    if dps_col and dps_col in df.columns:
        df['DPS'] = pd.to_numeric(df[dps_col], errors='coerce')
    elif operation_col and operation_col in df.columns:
        df['DPS'] = df[operation_col].apply(extract_dps)
    elif 'DPS' in df.columns:
        df['DPS'] = pd.to_numeric(df['DPS'], errors='coerce')
    else:
        raise ValueError("No DPS column found. Please select a DPS/Operation column, or provide a DPS column.")

    # choose steps column
    if smooth_col and smooth_col in df.columns:
        steps_source = smooth_col
    elif steps_col and steps_col in df.columns:
        steps_source = steps_col
    elif 'Smoothened Steps' in df.columns:
        steps_source = 'Smoothened Steps'
    elif 'Steps' in df.columns:
        steps_source = 'Steps'
    else:
        raise ValueError("No suitable steps column found. Please select Steps or Smoothened Steps.")

    df = df.dropna(subset=['DPS']).copy()
    df = df.sort_values('DPS').reset_index(drop=True)
    df['_steps_raw'] = pd.to_numeric(df[steps_source], errors='coerce').fillna(0.0)
    return df


def build_figure(df, params):
    """
    给定准备好的 df（含 DPS 和 _steps_raw）和参数字典 -> 返回 plotly Figure
    params: {use_smooth, smooth_window, smooth_poly, distance, prominence_factor, top_n}
    """
    x = df['DPS'].values
    y_raw = df['_steps_raw'].values

    # 平滑
    if params['use_smooth'] and len(y_raw) >= params['smooth_window'] and params['smooth_window'] % 2 == 1:
        try:
            y = savgol_filter(y_raw, params['smooth_window'], params['smooth_poly'])
        except Exception:
            y = y_raw.copy()
    else:
        y = y_raw.copy()

    # 自适应 prominence
    std_y = np.std(y) if len(y) > 0 else 0.0
    prominence_thresh = max(1e-6, params['prominence_factor'] * std_y)

    # 检测峰和谷
    det = detect_peaks_and_valleys(
        y,
        distance=max(1, int(params['distance'])),
        prominence=prominence_thresh
    )

    peaks_idx = det['peaks_idx']
    valleys_idx = det['valleys_idx']

    def top_n(indices, props, n):
        if len(indices) == 0 or n <= 0:
            return np.array([], dtype=int)
        prominences = props['prominences'] if 'prominences' in props else np.zeros(len(indices))
        order = np.argsort(prominences)[::-1][:n]
        return np.sort(indices[order])

    top_peaks = top_n(peaks_idx, det['peaks_props'], params['top_n'])
    top_valleys = top_n(valleys_idx, det['valleys_props'], params['top_n'])

    fig = go.Figure()

    # raw line
    fig.add_trace(
        go.Scatter(
            x=x, y=y_raw,
            mode='lines',
            name='Raw Steps',
            line=dict(width=1),
            opacity=0.25
        )
    )

    # smooth line
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Smooth Steps',
            line=dict(width=2.5)
        )
    )

    # all peaks/valleys
    if len(peaks_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=x[peaks_idx], y=y[peaks_idx],
                mode='markers',
                name='Peaks (all)',
                marker=dict(symbol='triangle-up', size=9, line=dict(width=1))
            )
        )

    if len(valleys_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=x[valleys_idx], y=y[valleys_idx],
                mode='markers',
                name='Valleys (all)',
                marker=dict(symbol='triangle-down', size=9, line=dict(width=1))
            )
        )

    # top peaks/valleys
    if len(top_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=x[top_peaks], y=y[top_peaks],
                mode='markers+text',
                name=f'Top {params["top_n"]} Peaks',
                marker=dict(symbol='triangle-up', size=15, color='red', line=dict(width=1)),
                text=[f"{v:.0f}" for v in y[top_peaks]],
                textposition='top center'
            )
        )

    if len(top_valleys) > 0:
        fig.add_trace(
            go.Scatter(
                x=x[top_valleys], y=y[top_valleys],
                mode='markers+text',
                name=f'Top {params["top_n"]} Valleys',
                marker=dict(symbol='triangle-down', size=15, color='blue', line=dict(width=1)),
                text=[f"{v:.0f}" for v in y[top_valleys]],
                textposition='bottom center'
            )
        )

    # surgery vertical line at DPS=0
    if len(y_raw) > 0:
        y_min = float(np.nanmin(np.concatenate([y_raw, y])))
        y_max = float(np.nanmax(np.concatenate([y_raw, y])))
    else:
        y_min, y_max = 0, 1

    fig.add_shape(
        type="line",
        x0=0, x1=0, y0=y_min, y1=y_max,
        line=dict(color="red", width=2, dash="dash")
    )

    fig.update_layout(
        title=params.get('title', 'Steps vs DPS'),
        xaxis_title='Days From Surgery (DPS)',
        yaxis_title='Steps',
        template='plotly_white',
        height=820,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    fig.update_xaxes(zeroline=False, showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# ---------------------------
# Dash 应用界面
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

UPLOAD_STYLE = {
    'width': '100%',
    'height': '72px',
    'lineHeight': '72px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderColor': '#d0d7de',
    'borderRadius': '14px',
    'textAlign': 'center',
    'backgroundColor': '#ffffff',
    'boxSizing': 'border-box',
    'cursor': 'pointer'
}

CARD_STYLE = {
    'borderRadius': '16px',
    'boxShadow': '0 1px 2px rgba(0,0,0,0.04)',
    'border': '1px solid #e5e7eb'
}

app.layout = dbc.Container(
    fluid=True,
    style={
        'padding': '16px 18px',
        'maxWidth': '1800px',
        'backgroundColor': '#f8f9fa'
    },
    children=[
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Peak/Valley Explorer",
                    style={'fontWeight': '600', 'marginBottom': '8px'}
                ),
                width=12
            ),
            className="mb-2"
        ),

        # 顶部区域：上传 + 选择文件/列
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Upload Excel", style={'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select an Excel File (.xlsx)')
                                    ]),
                                    style=UPLOAD_STYLE,
                                    multiple=False
                                ),
                                html.Div(id='upload-status', className='mt-2', style={'fontSize': '13px', 'color': '#4b5563'})
                            ]
                        ),
                        style=CARD_STYLE
                    ),
                    md=5
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Sheet name (optional):", style={'fontSize': '13px', 'fontWeight': '600'}),
                                                dcc.Input(
                                                    id='sheet-input',
                                                    type='text',
                                                    placeholder='Sheet1 (or leave blank)',
                                                    style={
                                                        'width': '100%',
                                                        'height': '38px',
                                                        'borderRadius': '10px',
                                                        'border': '1px solid #ced4da',
                                                        'padding': '0 12px'
                                                    }
                                                ),
                                            ],
                                            md=4
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("DPS / Operation column:", style={'fontSize': '13px', 'fontWeight': '600'}),
                                                dcc.Dropdown(
                                                    id='dps-column-dropdown',
                                                    placeholder='Select DPS/Operation column after upload'
                                                ),
                                            ],
                                            md=4
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Steps / Smoothened Steps column:", style={'fontSize': '13px', 'fontWeight': '600'}),
                                                dcc.Dropdown(
                                                    id='steps-column-dropdown',
                                                    placeholder='Select Steps column after upload'
                                                ),
                                            ],
                                            md=4
                                        ),
                                    ],
                                    className="g-3"
                                ),
                                html.Div(style={'height': '12px'}),
                                dbc.Button(
                                    "Generate/Refresh Plot",
                                    id='generate-button',
                                    color='primary',
                                    className='me-2'
                                ),
                                dbc.Button(
                                    "Download PNG",
                                    id='download-button',
                                    color='secondary'
                                ),
                                dcc.Download(id="download-image")
                            ]
                        ),
                        style=CARD_STYLE
                    ),
                    md=7
                )
            ],
            className="g-3 mb-3"
        ),

        # 主体区域：左侧参数栏 + 右侧大图
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Smoothing & Peak Controls", style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '12px'}),

                                dcc.Checklist(
                                    id='use-smooth',
                                    options=[{'label': ' Use Savitzky-Golay smoothing', 'value': 'on'}],
                                    value=['on'],
                                    style={'marginBottom': '8px'}
                                ),

                                html.Label("Smoothing window (odd integer):", style={'fontSize': '13px', 'fontWeight': '600'}),
                                dcc.Slider(
                                    id='smooth-window',
                                    min=3, max=51, step=2, value=11,
                                    marks={3: '3', 11: '11', 21: '21', 31: '31', 51: '51'}
                                ),
                                html.Div(style={'height': '18px'}),

                                html.Label("Smoothing poly order:", style={'fontSize': '13px', 'fontWeight': '600'}),
                                dcc.Slider(
                                    id='smooth-poly',
                                    min=2, max=5, step=1, value=3,
                                    marks={2: '2', 3: '3', 4: '4', 5: '5'}
                                ),
                                html.Hr(),

                                html.Label("Peak detection parameters:", style={'fontSize': '13px', 'fontWeight': '600'}),
                                html.Div(style={'height': '10px'}),

                                html.Label("Minimum distance between peaks:", style={'fontSize': '13px'}),
                                dcc.Slider(
                                    id='peak-distance',
                                    min=1, max=30, step=1, value=7,
                                    marks={1: '1', 5: '5', 10: '10', 20: '20', 30: '30'}
                                ),
                                html.Div(style={'height': '18px'}),

                                html.Label("Prominence factor (x std):", style={'fontSize': '13px'}),
                                dcc.Slider(
                                    id='prominence-factor',
                                    min=0.0, max=3.0, step=0.05, value=0.5,
                                    marks={0.0: '0', 0.5: '0.5', 1.0: '1.0', 2.0: '2.0'}
                                ),
                                html.Div(style={'height': '18px'}),

                                html.Label("Top N peaks/valleys:", style={'fontSize': '13px'}),
                                dcc.Slider(
                                    id='top-n',
                                    min=0, max=20, step=1, value=8,
                                    marks={0: '0', 5: '5', 10: '10', 20: '20'}
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, 'height': '100%'}
                    ),
                    md=3
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Loading(
                                    dcc.Graph(
                                        id='main-figure',
                                        figure=go.Figure(),
                                        style={'height': '82vh'}
                                    ),
                                    type="dot"
                                )
                            ],
                            style={'padding': '10px'}
                        ),
                        style={**CARD_STYLE, 'height': '100%'}
                    ),
                    md=9
                )
            ],
            className="g-3",
            style={'minHeight': 'calc(100vh - 180px)'}
        ),

        dcc.Store(id='stored-data', storage_type='memory'),
        html.Div(style={'height': '12px'})
    ]
)


# ---------------------------
# Callbacks
# ---------------------------

@app.callback(
    Output('upload-status', 'children'),
    Output('stored-data', 'data'),
    Output('dps-column-dropdown', 'options'),
    Output('steps-column-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('sheet-input', 'value'),
)
def handle_upload(contents, filename, sheet_name):
    if contents is None:
        return "No Uploaded File", None, [], []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if sheet_name and sheet_name.strip():
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=sheet_name)
        else:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return f"Excel Error: {e}", None, [], []

    cols = list(df.columns)
    opts = [{'label': c, 'value': c} for c in cols]
    data_json = df.to_json(date_format='iso', orient='split')
    msg = f"Uploaded: {filename} | Columns: {', '.join(cols[:10])}{'...' if len(cols) > 10 else ''}"
    return msg, data_json, opts, opts


@app.callback(
    Output('main-figure', 'figure'),
    Input('generate-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('dps-column-dropdown', 'value'),
    State('steps-column-dropdown', 'value'),
    State('use-smooth', 'value'),
    State('smooth-window', 'value'),
    State('smooth-poly', 'value'),
    State('peak-distance', 'value'),
    State('prominence-factor', 'value'),
    State('top-n', 'value'),
    prevent_initial_call=True
)
def update_plot(n_clicks, data_json, dps_col, steps_col,
                use_smooth, smooth_window, smooth_poly,
                peak_distance, prominence_factor, top_n):
    if data_json is None:
        fig = go.Figure()
        fig.update_layout(title="Please upload an Excel file first.")
        return fig

    df = pd.read_json(data_json, orient='split')

    try:
        df_prepared = prepare_df(df, operation_col=None, dps_col=dps_col, steps_col=steps_col)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Prepare Failure: {e}")
        return fig

    params = {
        'use_smooth': ('on' in (use_smooth or [])),
        'smooth_window': int(smooth_window) if smooth_window else 11,
        'smooth_poly': int(smooth_poly) if smooth_poly else 3,
        'distance': int(peak_distance) if peak_distance else 7,
        'prominence_factor': float(prominence_factor) if prominence_factor is not None else 0.5,
        'top_n': int(top_n) if top_n is not None else 8,
        'title': 'Uploaded patient'
    }

    fig = build_figure(df_prepared, params)

    pre_days = int(abs(df_prepared[df_prepared['DPS'] < 0]['DPS'].min())) if (df_prepared['DPS'] < 0).any() else 0
    post_days = int(df_prepared[df_prepared['DPS'] > 0]['DPS'].max()) if (df_prepared['DPS'] > 0).any() else 0
    fig.update_layout(
        title=f"Pre-op: {pre_days} days | Post-op: {post_days} days<br>Steps vs DPS",
        title_x=0.02
    )
    return fig


@app.callback(
    Output("download-image", "data"),
    Input("download-button", "n_clicks"),
    State('main-figure', 'figure'),
    prevent_initial_call=True
)
def download_png(n_clicks, figure):
    if figure is None:
        return None
    fig = go.Figure(figure)
    img_bytes = fig.to_image(format="png", width=1800, height=1000, scale=1)
    return dcc.send_bytes(lambda x: x.write(img_bytes), filename="steps_peaks_dps.png")


# Run
if __name__ == '__main__':
    app.run(debug=True)