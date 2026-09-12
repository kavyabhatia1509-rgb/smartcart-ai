import streamlit as st
st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PRODUCT DATA
# -----------------------------

products = {
    "product_id": ["P001", "P002", "P003", "P004", "P005",
                   "P006", "P007", "P008", "P009", "P010",
                   "P011", "P012", "P013", "P014", "P015",
                   "P016", "P017", "P018", "P019", "P020",
                   "P021", "P022", "P023", "P024", "P025",
                   "P026", "P027", "P028", "P029", "P030"],

    "product_name": [
        "Acer Aspire 5", "Lenovo IdeaPad Slim 5", "HP Pavilion 14",
        "ASUS VivoBook 15", "Dell Inspiron 15", "Samsung Galaxy A55",
        "OnePlus Nord CE", "Google Pixel 8a", "Redmi Note 14",
        "Samsung Galaxy S24 FE", "Sony WH-CH720N", "JBL Tune 770NC",
        "Boat Rockerz 550", "Anker Soundcore Q20i", "Realme Buds Wireless",
        "Redragon K552 Keyboard", "Logitech K380 Keyboard",
        "HP Wireless Keyboard", "Cosmic Byte Mechanical Keyboard",
        "Dell Multimedia Keyboard", "Logitech M331 Mouse", "HP Wireless Mouse",
        "Dell MS116 Mouse", "Redragon Gaming Mouse", "Lenovo Wireless Mouse",
        "LG 24-inch Monitor", "Samsung 24-inch Monitor",
        "Acer Nitro Monitor", "Lenovo 27-inch Monitor", "Dell 27-inch Monitor"
    ],

    "category": [
        "Laptop"] * 5 + ["Smartphone"] * 5 + ["Headphones"] * 5 +
        ["Keyboard"] * 5 + ["Mouse"] * 5 + ["Monitor"] * 5,

    "brand": [
        "Acer", "Lenovo", "HP", "ASUS", "Dell",
        "Samsung", "OnePlus", "Google", "Redmi", "Samsung",
        "Sony", "JBL", "Boat", "Anker", "Realme",
        "Redragon", "Logitech", "HP", "Cosmic Byte", "Dell",
        "Logitech", "HP", "Dell", "Redragon", "Lenovo",
        "LG", "Samsung", "Acer", "Lenovo", "Dell"
    ],

    "price": [
        55000, 58000, 62000, 52000, 60000,
        40000, 25000, 45000, 18000, 55000,
        8500, 6500, 3500, 4500, 2500,
        2500, 3500, 1800, 3000, 1500,
        1200, 1000, 800, 1800, 900,
        12000, 14000, 18000, 16000, 20000
    ],

    "features": [
        "laptop programming student performance",
        "laptop programming student productivity",
        "laptop programming office productivity",
        "laptop student office lightweight",
        "laptop programming business performance",
        "smartphone camera display performance",
        "smartphone 5g performance battery",
        "smartphone camera ai performance",
        "smartphone budget battery display",
        "smartphone camera performance display",
        "headphones wireless noise cancellation music",
        "headphones wireless noise cancellation bass",
        "headphones wireless gaming music bass",
        "headphones wireless noise cancellation music",
        "headphones wireless music lightweight",
        "keyboard mechanical gaming programming",
        "keyboard wireless compact productivity",
        "keyboard wireless office productivity",
        "keyboard mechanical gaming rgb",
        "keyboard office multimedia productivity",
        "mouse wireless office productivity",
        "mouse wireless office laptop",
        "mouse wired office productivity",
        "mouse gaming performance rgb",
        "mouse wireless laptop productivity",
        "monitor display office productivity",
        "monitor display office entertainment",
        "monitor gaming display performance",
        "monitor large display productivity",
        "monitor large display office productivity"
    ]
}

df_products = pd.DataFrame(products)

st.title("🛒 SmartCart AI")
st.write("Your AI-powered personalized shopping assistant")

# -----------------------------
# USER INTERACTION DATA
# -----------------------------

interactions = {
    "user_id": [
        "U001", "U001", "U001",
        "U002", "U002", "U002",
        "U003", "U003", "U003",
        "U004", "U004", "U004",
        "U005", "U005", "U005"
    ],

    "product_id": [
        "P001", "P003", "P016",
        "P006", "P011", "P012",
        "P021", "P022", "P025",
        "P026", "P027", "P029",
        "P002", "P004", "P018"
    ],

    "interaction": [
        "purchased", "liked", "liked",
        "purchased", "liked", "viewed",
        "purchased", "liked", "liked",
        "purchased", "liked", "viewed",
        "liked", "viewed", "liked"
    ]
}

df_interactions = pd.DataFrame(interactions)

interaction_weights = {
    "viewed": 1,
    "liked": 2,
    "purchased": 3
}

df_interactions["weight"] = df_interactions["interaction"].map(
    interaction_weights
)

df_user_products = pd.merge(
    df_interactions,
    df_products,
    on="product_id"
)

# -----------------------------
# ML RECOMMENDATION ENGINE
# -----------------------------

# Convert product features into TF-IDF vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    df_products["features"]
)

# Calculate similarity between all products
similarity_matrix = cosine_similarity(
    tfidf_matrix
)


def recommend_for_user(user_id, top_n=5):

    # Get the user's previous interactions
    user_data = df_user_products[
        df_user_products["user_id"] == user_id
    ]

    # Get products and interaction weights
    user_products = user_data["product_id"].tolist()
    user_weights = user_data["weight"].values

    # Convert product IDs to dataframe indexes
    user_indices = [
        df_products.index[
            df_products["product_id"] == product_id
        ][0]
        for product_id in user_products
    ]

    # Get similarity scores
    user_similarity_scores = similarity_matrix[user_indices]

    # Give more importance to stronger interactions
    weighted_scores = (
        user_similarity_scores * user_weights[:, None]
    )

    # Combine all influence scores
    final_scores = weighted_scores.sum(axis=0)

    # Rank products
    ranked_indices = final_scores.argsort()[::-1]

    # Remove products the user already interacted with
    filtered_indices = [
        index for index in ranked_indices
        if df_products.iloc[index]["product_id"]
        not in user_products
    ]

    # Select top recommendations
    top_indices = filtered_indices[:top_n]

    recommendations = df_products.iloc[
        top_indices
    ].copy()

    # Add recommendation score
    recommendations["recommendation_score"] = [
        round(final_scores[index], 2)
        for index in top_indices
    ]

    return recommendations[
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "recommendation_score"
        ]
    ]

# -----------------------------
# RECOMMENDATION EXPLANATION
# -----------------------------

def explain_recommendation(user_id, recommended_product_id):

    user_data = df_user_products[
        df_user_products["user_id"] == user_id
    ]

    user_products = user_data["product_id"].tolist()
    user_weights = user_data["weight"].values

    user_indices = [
        df_products.index[
            df_products["product_id"] == product_id
        ][0]
        for product_id in user_products
    ]

    recommended_index = df_products.index[
        df_products["product_id"] == recommended_product_id
    ][0]

    influence_scores = (
        similarity_matrix[user_indices, recommended_index]
        * user_weights
    )

    max_index = influence_scores.argmax()

    influencing_product = user_data.iloc[
        max_index
    ]["product_name"]

    recommended_product = df_products.iloc[
        recommended_index
    ]["product_name"]

    return (
        f"Recommended because you interacted with "
        f"{influencing_product}, which is similar to "
        f"{recommended_product}."
    )

# -----------------------------

# STREAMLIT WEBSITE INTERFACE

st.title("🛒 SmartCart AI")
st.subheader("AI-Powered Personalized Shopping Assistant")

st.write(
    "SmartCart AI analyzes your previous interactions "
    "and recommends products that match your interests."
)

st.divider()

user_id = st.selectbox(
    "👤 Select User",
    ["U001", "U002", "U003", "U004", "U005"]
)

category = st.selectbox(
    "📂 Select Category",
    ["All", "Laptop", "Smartphone", "Headphones",
     "Keyboard", "Mouse", "Monitor"]
)

budget = st.slider(
    "💰 Maximum Budget (₹)",
    min_value=1000,
    max_value=60000,
    value=60000,
    step=1000
)

top_n = st.slider(
    "🔢 Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🤖 Get Recommendations"):

    recommendations = recommend_for_user(user_id, 30)

    if category != "All":
        recommendations = recommendations[
            recommendations["category"] == category
        ]

    recommendations = recommendations[
        recommendations["price"] <= budget
    ]

    recommendations = recommendations.head(top_n)

    st.subheader("✨ AI Recommendations")

    if len(recommendations) == 0:

        st.warning(
            "No products match your selected category and budget."
        )

    else:

        for _, product in recommendations.iterrows():

            st.markdown(
                f"### 🛍️ {product['product_name']}"
            )

            st.write(
                f"**Category:** {product['category']}"
            )

            st.write(
                f"**Price:** ₹{product['price']:,}"
            )

            st.write(
                f"**Recommendation Score:** "
                f"{product['recommendation_score']}"
            )

            explanation = explain_recommendation(
                user_id,
                product["product_id"]
            )

            st.info(
                f"💡 {explanation}"
            )

            st.divider()
