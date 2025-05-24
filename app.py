import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Page config ----
st.set_page_config(
    page_title="Superstore Profit Classifier",
    page_icon="📈",
    layout="wide"
)

# ---- Load model & features ----
@st.cache_data
def load_model_and_features():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_list = load_model_and_features()

# ---- Load data ----
@st.cache_data
def load_data():
    df = pd.read_csv('data/superstore_cleaned.csv')
    # Ensure high_profit exists
    if 'high_profit' not in df.columns:
        df['high_profit'] = (df['profit'] > 100).astype(int)
    return df

df = load_data()

# ---- Sidebar Filters ----
st.sidebar.header("🔎 Filter Data Preview")

region_filter = st.sidebar.multiselect(
    "Select Region(s)",
    options=df["region"].unique(),
    default=df["region"].unique()
)

product_cat_filter = st.sidebar.multiselect(
    "Select Product Category(ies)",
    options=df["product_category"].unique(),
    default=df["product_category"].unique()
)

filtered_df = df[
    (df["region"].isin(region_filter)) &
    (df["product_category"].isin(product_cat_filter))
]

# ---- Title ----
st.title("📊 Superstore Profit Classifier")
st.markdown("""
Predict whether an order will generate **High Profit (Profit > $100)**  
Fill in order details and get an instant classification result! ⚡
""")

# ---- Input Section ----
st.header("🧾 Enter Order Details for Prediction")

sales = st.number_input("💰 Sales ($)", min_value=0.0, value=100.0, step=10.0)
order_quantity = st.number_input("📦 Order Quantity", min_value=1, value=5)
discount = st.number_input("🏷️ Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
shipping_cost = st.number_input("🚚 Shipping Cost ($)", min_value=0.0, value=15.0)

product_category = st.text_input("📦 Product Category (exact name)", value="Furniture")
product_sub_category = st.text_input("📋 Product Sub-category (exact name)", value="Chairs")
region = st.text_input("📍 Region (exact name)", value="West")

# ---- Prepare input for model ----
input_data = {feat: 0 for feat in feature_list}
input_data['sales'] = sales
input_data['order_quantity'] = order_quantity
input_data['discount'] = discount
input_data['shipping_cost'] = shipping_cost

# One-hot encode matching columns
for col in feature_list:
    if col == f"product_category_{product_category}":
        input_data[col] = 1
    if col == f"product_sub-category_{product_sub_category}":
        input_data[col] = 1
    if col == f"region_{region}":
        input_data[col] = 1

input_df = pd.DataFrame([input_data])

# ---- Prediction ----
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df[feature_list])[0]
        proba = model.predict_proba(input_df[feature_list])[0][1]

        if prediction == 1:
            st.success(f"✅ High Profit predicted! (Confidence: {proba:.2%}) 🎉")
        else:
            st.info(f"ℹ️ Not High Profit. (Confidence: {1 - proba:.2%})")

        # Download result
        result_df = input_df.copy()
        result_df['predicted_high_profit'] = prediction
        result_df['confidence'] = proba
        st.download_button(
            label="📥 Download Prediction as CSV",
            data=result_df.to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"🚫 Error in prediction: {e}")

# ---- Data Preview ----
st.markdown("---")
st.header("📋 Filtered Superstore Data Preview")
st.write(f"Showing {len(filtered_df)} records:")
st.dataframe(filtered_df.head(100).style.background_gradient(cmap="Blues"))

# ---- Visualizations ----
st.markdown("---")
st.header("📈 Visual Insights")

plot_df = filtered_df.sample(min(len(filtered_df), 1000), random_state=42)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profit Distribution")
    fig, ax = plt.subplots()
    sns.histplot(plot_df['profit'], bins=30, kde=True, ax=ax, color='green')
    st.pyplot(fig)

with col2:
    st.subheader("Sales vs Discount")
    fig, ax = plt.subplots()
    if 'high_profit' in plot_df.columns:
        sns.scatterplot(data=plot_df, x='sales', y='discount', hue='high_profit', palette='coolwarm', ax=ax)
    else:
        sns.scatterplot(data=plot_df, x='sales', y='discount', color='blue', ax=ax)
    st.pyplot(fig)

# ---- Footer ----
st.markdown("---")
st.markdown("""
Made with ❤️ by **Ayrin Akter Supty**  
Stay creative and keep coding! 🚀
""")
