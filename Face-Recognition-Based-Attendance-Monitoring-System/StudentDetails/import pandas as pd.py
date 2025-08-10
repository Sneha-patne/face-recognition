import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# --- 1. Load and Clean Data ---
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    return df

# --- 2. Basic Sales Insights ---
def basic_sales_insights(df):
    total_sales = df['TotalSales'].sum()
    top_regions = df.groupby('Region')['TotalSales'].sum().sort_values(ascending=False)
    top_products = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(5)
    return total_sales, top_regions, top_products

# --- 3. Visualizations ---
def plot_sales_by_region(df):
    region_data = df.groupby('Region')['TotalSales'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Region', y='TotalSales', data=region_data, ax=ax)
    ax.set_title('Sales by Region')
    st.pyplot(fig)

def plot_top_products(df):
    product_data = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(5).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='TotalSales', y='Product', data=product_data, ax=ax)
    ax.set_title('Top 5 Products')
    st.pyplot(fig)

# --- 4. Predictive Sales Model ---
def train_sales_model(df):
    df['Month'] = df['Date'].dt.month
    X = df[['Quantity', 'UnitPrice', 'Month']]
    y = df['TotalSales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

# --- 5. Streamlit Dashboard ---
def run_dashboard():
    st.title("Consumer Goods Company - Sales Insights")
    df = load_and_clean_data("sales_data.csv")

    st.subheader("Raw Data")
    st.dataframe(df.head())

    total_sales, top_regions, top_products = basic_sales_insights(df)

    st.metric("Total Sales", f"${total_sales:,.2f}")
    st.subheader("Sales by Region")
    plot_sales_by_region(df)

    st.subheader("Top 5 Products")
    plot_top_products(df)

    st.subheader("Train Predictive Model")
    model, score = train_sales_model(df)
    st.success(f"Model R^2 Score: {score:.2f}")

# --- Run the App ---
if __name__ == "__main__":
    run_dashboard()
