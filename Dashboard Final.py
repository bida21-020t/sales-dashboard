import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score

# Load dataset
df = pd.read_csv('web_interaction_data_with_clusters.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['DateOnly'] = df['Timestamp'].dt.date  # Create DateOnly column once upfront

# Cluster label mapping
CLUSTER_LABELS = {
    0: "Demo Request",
    1: "Virtual Assistant Request",
    2: "Prototype Request",
    3: "Schedule Demo",
    4: "Quote Request"
}

# === Precompute Evaluation Metrics ===
demo_df = df[df['Requested Demo'].astype(str).str.strip().str.lower().isin(['yes', 'y', 'true', '1'])].copy()
daily_counts = demo_df.groupby('DateOnly').size().reset_index(name='Sales')
q1 = daily_counts['Sales'].quantile(0.33)
q2 = daily_counts['Sales'].quantile(0.66)

daily_counts['Pseudo_Label'] = daily_counts['Sales'].apply(lambda x: 0 if x < q1 else (1 if x < q2 else 2))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(daily_counts[['Sales']])
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

mapping = {}
for cluster in np.unique(cluster_labels):
    true_labels = daily_counts['Pseudo_Label'][cluster_labels == cluster]
    majority_label = true_labels.value_counts().idxmax()
    mapping[cluster] = majority_label
mapped_preds = [mapping[label] for label in cluster_labels]

accuracy_score_value = accuracy_score(daily_counts['Pseudo_Label'], mapped_preds)
silhouette_score_value = silhouette_score(scaled_data, cluster_labels)
db_index_value = davies_bouldin_score(scaled_data, cluster_labels)

# === Precompute Anomaly Chart ===
filtered_df = df[df['Product Sold'] == 1].copy()
daily_sales = filtered_df.groupby('DateOnly').size().reset_index(name='Sales')
daily_sales['RollingMean'] = daily_sales['Sales'].rolling(window=7, center=True).mean()
daily_sales['RollingStd'] = daily_sales['Sales'].rolling(window=7, center=True).std()
daily_sales['Zscore'] = (daily_sales['Sales'] - daily_sales['RollingMean']) / daily_sales['RollingStd']
daily_sales['Anomaly'] = daily_sales['Zscore'].abs() > 1.5

anomaly_fig = go.Figure()
anomaly_fig.add_trace(go.Scatter(x=daily_sales['DateOnly'], y=daily_sales['Sales'],
                  mode='lines+markers', name='Daily Sales'))
anomaly_fig.add_trace(go.Scatter(x=daily_sales['DateOnly'], y=daily_sales['RollingMean'],
                  mode='lines', name='Rolling Mean'))
anomaly_fig.add_trace(go.Scatter(x=daily_sales.loc[daily_sales['Anomaly'], 'DateOnly'],
                  y=daily_sales.loc[daily_sales['Anomaly'], 'Sales'],
                  mode='markers', name='Anomalies',
                  marker=dict(color='red', size=10)))
anomaly_fig.update_layout(title='Anomaly Detection in Daily Product Sales',
                     xaxis_title='Date', yaxis_title='Sales')

# Initialize app with callback exception suppression
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Sales Dashboard"

# Filters
available_regions = sorted(df['Country'].dropna().unique())
available_products = sorted(df['Product'].dropna().unique())
available_salespeople = sorted(df['Salesperson'].dropna().unique())

# App Layout 
app.layout = html.Div(style={'height': '100vh', 'overflow': 'hidden'}, children=[
    dbc.Container(fluid=True, style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'}, children=[
        html.H1("AI Solution Sales Dashboard", className="text-center mb-2", style={'flex': '0 0 auto'}),
        
        dbc.Tabs([
            dbc.Tab(label="Sales Executive", tab_id="sales_exec"),
            dbc.Tab(label="Sales Manager", tab_id="sales_mgr"),
            dbc.Tab(label="Business Analyst", tab_id="biz_analyst"),
            dbc.Tab(label="Regional Coordinator", tab_id="regional_coord")
        ], id="tabs", active_tab="sales_exec", style={'flex': '0 0 auto'}),

        html.Div(style={'flex': '1 1 auto', 'overflow': 'hidden'}, children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id="date-filter",
                        min_date_allowed=df['Timestamp'].min().date(),
                        max_date_allowed=df['Timestamp'].max().date(),
                        start_date=df['Timestamp'].min().date(),
                        end_date=df['Timestamp'].max().date(),
                        style={'width': '100%'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Region (Country)"),
                    dcc.Dropdown(
                        id="region-filter",
                        options=[{"label": x, "value": x} for x in available_regions],
                        multi=True,
                        style={'width': '100%'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Product"),
                    dcc.Dropdown(
                        id="product-filter",
                        options=[{"label": x, "value": x} for x in available_products],
                        multi=True,
                        style={'width': '100%'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Salesperson"),
                    dcc.Dropdown(
                        id="salesperson-filter",
                        options=[{"label": x, "value": x} for x in available_salespeople],
                        multi=True,
                        style={'width': '100%'}
                    )
                ], width=3)
            ], className="mb-2"),

            dbc.Button("Export to CSV", id="export-csv", color="primary", className="mb-2 me-2"),
            dcc.Download(id="download-csv"),

            html.Div(id="tab-content", style={'height': 'calc(100vh - 250px)'})
        ])
    ])
])

# ================= Layout Functions =================
def get_sales_exec_layout(dff):
    # Get actual sales data from 'Product Sold' column
    sales_data = dff[dff['Product Sold'].astype(str).str.strip().str.lower().isin(['yes', 'y', 'true', '1'])].copy()
    # Get demo requests
    demo_requests = dff[dff['Requested Demo'].astype(str).str.strip().str.lower().isin(['yes', 'y', 'true', '1'])].copy()
    
    total_visits = len(dff)
    total_sales = len(sales_data)
    total_demos = len(demo_requests)
    
    # Calculate sales conversion rate for the gauge
    if 'Product Sold' in dff.columns and not dff.empty:
        sales_conversion_rate = round(total_sales / total_visits * 100, 2) if total_visits > 0 else 0
    else:
        sales_conversion_rate = 0
    
    # Target is now 30% for sales conversion rate
    target_percentage = 30
    target_met = sales_conversion_rate >= target_percentage
    color = "success" if target_met else "danger"
    
    # Create gauge chart for sales conversion rate
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sales_conversion_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sales Conversion Rate (%)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#FF00FF"},  # Vibrant magenta
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, target_percentage], 'color': "#FFD700"},  # Gold
                {'range': [target_percentage, 100], 'color': "#32CD32"}],  # Lime green
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sales_conversion_rate}
        }
    ))
    gauge_fig.update_layout(height=300)

    # Create figure only if there's data
    if not sales_data.empty:
        fig = px.histogram(sales_data, x="Product", color="Country", barmode="group",
                          title="Sales by Product and Country")
        fig.update_layout(height=300)
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Sales by Product and Country - No Data Available",
            height=300,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No sales data available for selected filters",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Total Visits", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                dbc.CardBody(html.Div(f"{total_visits:,}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
            ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Total Sales", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                dbc.CardBody(html.Div(f"{total_sales:,}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
            ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Demo Requests", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                dbc.CardBody(html.Div(f"{total_demos:,}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
            ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Sales Target", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                dbc.CardBody(html.Div("Met" if target_met else "Below",
                                    className=f"text-{color}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
            ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=3)
        ], className="mb-3", style={'height': '100px'}),
        
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, style={'height': '300px'})),  # Bar chart on the left
            dbc.Col(dcc.Graph(figure=gauge_fig, style={'height': '300px'}))  # Gauge on the right
        ], className="mb-2"),
    ], style={'height': '100%'})

def get_sales_mgr_layout(dff):
    job_counts = dff.groupby(['Salesperson', 'Job Type']).size().reset_index(name='Count')
    
    if not job_counts.empty:
        fig = px.bar(job_counts, x='Salesperson', y='Count', color='Job Type', barmode='stack',
                    title="Job Requests Breakdown by Salesperson")
        fig.update_layout(height=400)
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Job Requests Breakdown by Salesperson - No Data Available",
            height=400,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No job request data available for selected filters",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
    
    return html.Div([
        html.H4("Sales Manager View", className="mb-2"),
        dcc.Graph(figure=fig, style={'height': '400px'})
    ], style={'height': '100%'})

def get_biz_analyst_layout(dff):
    return html.Div([
        html.H4("Business Analyst View", className="mb-2"),
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="biz_overview"),
            dbc.Tab(label="Performance & Anomalies", tab_id="biz_metrics"),
        ], id="biz-tabs", active_tab="biz_overview"),
        html.Div(id="biz-tab-content", style={'height': 'calc(100vh - 350px)', 'overflow': 'auto'})
    ])

def get_regional_coord_layout(dff):
    # Filter for actual sales data
    sales_data = dff[dff['Product Sold'].astype(str).str.strip().str.lower().isin(['yes', 'y', 'true', '1'])].copy()
    
    if 'Country' in sales_data.columns and not sales_data.empty:
        heat_data = sales_data.groupby('Country').size().reset_index(name='Sales')
        fig = px.choropleth(heat_data, locations='Country', locationmode='country names',
                           color='Sales', title='Sales Distribution by Country',
                           color_continuous_scale='Viridis')
        fig.update_layout(height=500)
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Sales Distribution by Country - No Data Available",
            height=500,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No sales data available for selected filters",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
    
    return html.Div([
        html.H4("Regional Coordinator View", className="mb-2"),
        dcc.Graph(figure=fig, style={'height': '400px'})
    ], style={'height': '90%'})

# ================= Callbacks =================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("region-filter", "value"),
    Input("product-filter", "value"),
    Input("salesperson-filter", "value")
)
def update_tab(tab, start_date, end_date, region, product, salespeople):
    dff = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]
    if region:
        dff = dff[dff['Country'].isin(region)]
    if product:
        dff = dff[dff['Product'].isin(product)]
    if salespeople:
        dff = dff[dff['Salesperson'].isin(salespeople)]

    if tab == "sales_exec":
        return get_sales_exec_layout(dff)
    elif tab == "sales_mgr":
        return get_sales_mgr_layout(dff)
    elif tab == "biz_analyst":
        return get_biz_analyst_layout(dff)
    elif tab == "regional_coord":
        return get_regional_coord_layout(dff)
    return html.P("Tab not found")

@app.callback(
    Output("biz-tab-content", "children"),
    Input("biz-tabs", "active_tab"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("region-filter", "value"),
    Input("product-filter", "value"),
    Input("salesperson-filter", "value")
)
def update_biz_tab(active_tab, start_date, end_date, region, product, salespeople):
    # Default to "biz_overview" if no tab is active
    if active_tab is None:
        active_tab = "biz_overview"
        
    dff = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]
    if region:
        dff = dff[dff['Country'].isin(region)]
    if product:
        dff = dff[dff['Product'].isin(product)]
    if salespeople:
        dff = dff[dff['Salesperson'].isin(salespeople)]
    
    # ====== Action Percentage Pie ======
    total_users = len(dff)
    requested_demo = dff['Requested Demo'].astype(str).str.strip().str.lower().isin(['yes', '1', 'true']).sum()
    product_sold = dff['Product Sold'].astype(int).sum()
    both = ((dff['Requested Demo'].astype(str).str.strip().str.lower().isin(['yes', '1', 'true'])) & 
            (dff['Product Sold'].astype(int) == 1)).sum()
    neither = total_users - requested_demo - product_sold + both  # prevent double counting

    action_data = pd.DataFrame({
        'Action': ['Requested Demo Only', 'Purchased Only', 'Both Actions', 'No Action'],
        'Count': [requested_demo - both, product_sold - both, both, neither]
    })

    action_fig = px.pie(action_data, names='Action', values='Count', title='Key Action Distribution (%)',
                        hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
    
    action_graph = dbc.Row([
        dbc.Col(dcc.Graph(figure=action_fig, style={'height': '350px'}))
    ], className="mt-4")
    
    conversions = dff['Converted'].astype(str).str.strip().str.lower().isin(['yes', 'y', 'true', '1'])
    conversion_rate = round(conversions.mean() * 100, 2) if not conversions.empty else 0

    top_products = dff['Product'].value_counts().reset_index()
    top_products.columns = ['Product', 'Count']
    top_product_name = top_products.iloc[0]['Product'] if not top_products.empty else "N/A"

    # Cluster summary with meaningful labels - fixed version
    dff = dff.copy()  # Ensure we're working with a copy
    dff.loc[:, 'Cluster_Label'] = dff['Cluster'].map(CLUSTER_LABELS)
    cluster_summary = dff['Cluster_Label'].value_counts().reset_index()
    cluster_summary.columns = ['Cluster', 'Count']
    
    # Create cluster figure with numeric x-axis labels only and no numbers inside bars
    if not cluster_summary.empty:
        cluster_fig = go.Figure()
        for i, row in cluster_summary.iterrows():
            cluster_fig.add_trace(go.Bar(
                x=[row['Cluster']],
                y=[row['Count']],
                name=row['Cluster'],
                textposition='none'  # This removes the numbers inside the bars
            ))
        cluster_fig.update_layout(
            title="Customer Count per Cluster",
            height=400,
            xaxis={
                'tickvals': [0, 1, 2, 3, 4],
                'ticktext': ['0', '1', '2', '3', '4'],
                'title': 'Cluster'
            },
            yaxis={'title': 'Count'},
            showlegend=True
        )
    else:
        cluster_fig = go.Figure()

    products_fig = px.pie(top_products.head(5), names='Product', values='Count',
                        title="Top 5 Products by Request", height=400) if not top_products.empty else go.Figure()

    if active_tab == "biz_overview":
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Conversion Rate", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                    dbc.CardBody(html.Div(f"{conversion_rate}%", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
                ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Top Product", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                    dbc.CardBody(html.Div(top_product_name, style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
                ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=6)
            ], className="mb-3", style={'height': '100px'}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=products_fig, style={'height': '400px'}), width=6),
                dbc.Col(dcc.Graph(figure=cluster_fig, style={'height': '400px'}), width=6)
            ], style={'height': '400px'}),
            action_graph
        ], style={'overflow': 'auto'})
    
    elif active_tab == "biz_metrics":
        return html.Div([
            html.H5("Model Evaluation Metrics", className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Silhouette Score", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                    dbc.CardBody(html.Div(f"{silhouette_score_value:.4f}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
                ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Davies-Bouldin Index", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                    dbc.CardBody(html.Div(f"{db_index_value:.4f}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
                ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Accuracy Score", style={'padding': '0.5rem', 'font-size': '0.9rem', 'background-color': '#f8f9fa'}),
                    dbc.CardBody(html.Div(f"{accuracy_score_value:.4f}", style={'margin': '0', 'font-size': '1rem', 'text-align': 'center'}))
                ], style={'height': '100%', 'padding': '0.25rem', 'border': '1px solid #dee2e6', 'border-radius': '5px'}), width=4),
            ], className="mb-4"),
            html.H5("Anomaly Detection in Product Sales", className="mt-3"),
            dcc.Graph(figure=anomaly_fig, style={'height': '500px'})
        ], style={'overflow': 'auto'})
    return "Tab content not found"

@app.callback(
    Output("download-csv", "data"),
    Input("export-csv", "n_clicks"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("region-filter", "value"),
    State("product-filter", "value"),
    State("salesperson-filter", "value"),
    prevent_initial_call=True
)
def export_filtered_data(n_clicks, start_date, end_date, region, product, salespeople):
    dff = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]
    if region:
        dff = dff[dff['Country'].isin(region)]
    if product:
        dff = dff[dff['Product'].isin(product)]
    if salespeople:
        dff = dff[dff['Salesperson'].isin(salespeople)]
    return dcc.send_data_frame(dff.to_csv, "filtered_dashboard_data.csv")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)