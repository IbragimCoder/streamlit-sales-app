import streamlit as st
import pandas as pd
import joblib

# @st.cache_data говорит Streamlit: "Выполни эту функцию один раз и запомни результат"
@st.cache_data
def load_data():
    data = pd.read_csv('last_satis.csv')
    data = data.drop(columns = 'Unnamed: 0')
    data['Tarix'] = pd.to_datetime(data['Tarix'], unit='D', origin='1899-12-30')
    data['Tarix'] = data['Tarix'].dt.strftime('%Y-%m')
    return data

# @st.cache_resource говорит: "Загрузи этот ресурс (модель) один раз и держи в памяти"
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        model = joblib.load(f)
    return model

try:
    data = load_data()
    model = load_model('regressor_pipeline.joblib')
except FileNotFoundError:
    st.error("Error")
    st.stop() # Останавливаем выполнение скрипта, если файлы не найдены


st.title('Dashboard and sales prediction 📈')
st.write("---")

st.header("🤖 Predict the sale amount")

col1, col2 = st.columns(2)

with col1:
    selected_store = st.selectbox("`Mağaza`", options=sorted(data['Mağaza'].unique()))
    selected_product_name = st.selectbox("`Məhsul_adi`", options=sorted(data['Məhsul_adi'].unique()))
    selected_card_id = st.number_input("`Kart_nomresi`", min_value=0, value=1000, step=1)
    
with col2:
    selected_product_id = st.number_input("`Məhsul_nomresi`", min_value=0, value=1, step=1)
    selected_quantity = st.number_input("`Məhsul sayi`", min_value=1, value=1, step=1)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'Mağaza': [selected_store],
        'Kart_nomresi': [selected_card_id],
        'Məhsul_nomresi': [selected_product_id],
        'Məhsul_adi': [selected_product_name],
        'Məhsul sayi': [selected_quantity]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted sale amount: ${prediction[0]:.2f}")

st.write("---")

st.header("📊 Analytics")
st.write("### Key business indicators")

total_sales = data['Ümumi satış'].sum()
total_items_sold = data['Məhsul sayi'].sum()
unique_customers = data['Kart_nomresi'].nunique()

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Total revenue", f"{total_sales:,.0f} $") # revenue - выручка
m_col2.metric("Total items sold", f"{total_items_sold:,.0f}")
m_col3.metric("Unique clients", f"{unique_customers:,.0f}")

st.write("### Sales Leaders")

top_5_products = data.groupby('Məhsul_adi')['Ümumi satış'].sum().nlargest(5)
top_5_stores = data.groupby('Mağaza')['Ümumi satış'].sum().nlargest(5)

g_col1, g_col2 = st.columns(2)
with g_col1:
    st.write("Top 5 products by revenue")
    st.bar_chart(top_5_products)
with g_col2:
    st.write("Top 5 stores by revenue")
    st.bar_chart(top_5_stores)

st.write("---") 
st.write("### In-depth product analysis")

selected_product_for_analysis = st.selectbox(
    "Select a product for detailed analysis:",
    options=sorted(data['Məhsul_adi'].unique()),
    key='product_analysis_selectbox'
)

st.write(f"#### Detailed product Information: '{selected_product_for_analysis}'")
product_data = data[data['Məhsul_adi'] == selected_product_for_analysis]
st.dataframe(product_data)

st.write(f"#### Rating of stores by product sales")
sales_by_store = product_data.groupby('Mağaza')['Ümumi satış'].sum().sort_values(ascending=False)

if not sales_by_store.empty:
    st.bar_chart(sales_by_store)
else:
    st.warning("There is no sales data to display")