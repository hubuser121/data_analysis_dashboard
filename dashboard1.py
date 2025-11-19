import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = pd.read_csv('titanic.csv')

app.layout = html.Div([
    html.H1("Titanic Data Analysis Dashboard"),
    dcc.Graph(figure=px.histogram(df, x='Age', title='Age Distribution')),
    dcc.Graph(figure=px.bar(df.groupby('Pclass')['Survived'].mean().reset_index(), x='Pclass', y='Survived', title='Survival by Class')),
    dcc.Graph(figure=px.scatter(df, x='Age', y='Fare', title='Age vs Fare')),
    dcc.Graph(figure=px.imshow(df.select_dtypes(include='number').corr(), title='Correlation Matrix')),
    dcc.Graph(figure=px.box(df, x='Pclass', y='Age', title='Age by Class')),
    dcc.Graph(figure=px.violin(df, x='Sex', y='Age', title='Age by Sex')),
])

if __name__ == '__main__':
    app.run(debug=True)
