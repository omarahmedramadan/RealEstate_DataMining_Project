import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM

# --- 1. Page Configuration ---
st.set_page_config(page_title="NYC Airbnb Real Estate Dashboard", layout="wide")

# --- 2. Load and Clean Data ---
@st.cache_data
def load_data():
    # Use the relative path established in previous steps
    path = "RealEstate_DataMining_Project_code/airbnb_cleaned.csv"
    df = pd.read_csv(path)
    
    # Basic cleaning consistent with the Jupyter Notebook
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # Remove extreme outliers (Keep 95% of data) for better visualization and modeling
    q = df['price'].quantile(0.95)
    df = df[(df['price'] > 0) & (df['price'] < q)]
    return df

df = load_data()

# --- 3. Sidebar Filters ---
st.sidebar.header("Global Filters")
neighborhood = st.sidebar.multiselect(
    "Select Neighborhood Group:",
    options=df["neighbourhood_group"].unique(),
    default=df["neighbourhood_group"].unique()
)

# Apply filters to the main dataframe
mask = df["neighbourhood_group"].isin(neighborhood)
df_filtered = df[mask]

# --- 4. Main Header and KPIs ---
st.title("🏠 NYC Real Estate Data Mining Dashboard")
st.markdown("Developed by: **Omar Ahmed Ramadan** & **Ahmed Mohamed Fathy**")

# Statistics Section (KPIs)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Listings", f"{len(df_filtered):,}")
with col2:
    st.metric("Avg Price", f"${df_filtered['price'].mean():.2f}")
with col3:
    # KPI showing the 5 Neighborhood Groups
    st.metric("Neighborhood Groups", f"{df['neighbourhood_group'].nunique()}")
with col4:
    # KPI showing the 3 Room Types
    st.metric("Room Types", f"{df['room_type'].nunique()}")

# --- 5. Interactive Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Market Prediction Tool", 
    "🗺️ Market Segmentation", 
    "📊 Price Categories", 
    "🚨 Anomaly Detection"
])

# --- TAB 1: Real-Time Price Predictor (Regression) ---
with tab1:
    st.header("Price Predictor Tool (Linear Regression)")
    st.info("Input property details below to estimate the market price.")
    
    # Layout for user input
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        user_ng = st.selectbox("Neighborhood Group", df['neighbourhood_group'].unique())
        user_rt = st.selectbox("Room Type", df['room_type'].unique())
        user_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=1)
        
    with input_col2:
        user_reviews = st.number_input("Number of Reviews", min_value=0, max_value=500, value=10)
        user_avail = st.slider("Availability (Days per Year)", 0, 365, 150)
        user_host_count = st.number_input("Host Listing Count", 1, 50, 1)

    # Simple training of the model on the fly for prediction
    # Preprocessing categorical data for the model
   # Preprocessing categorical data for the model
    df_model = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)
    
    # Drop all text and unnecessary columns ('name' and 'host_name' added here)
    X = df_model.drop(columns=['price', 'id', 'name', 'host_name', 'host_id', 'latitude', 'longitude', 'neighbourhood', 'last_review'], errors='ignore')
    
    # Force X to only keep numeric columns to prevent ValueError
    X = X.select_dtypes(include=['number', 'bool'])
    y = df_model['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Prepare the user input for prediction
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0 # Initialize with zeros
    input_data['minimum_nights'] = user_nights
    input_data['number_of_reviews'] = user_reviews
    input_data['availability_365'] = user_avail
    input_data['calculated_host_listings_count'] = user_host_count
    
    # Set the dummy variables based on selection
    if f"neighbourhood_group_{user_ng}" in input_data.columns:
        input_data[f"neighbourhood_group_{user_ng}"] = 1
    if f"room_type_{user_rt}" in input_data.columns:
        input_data[f"room_type_{user_rt}"] = 1
        
    # Final Prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Price: ${prediction:.2f} per night")
    
    # Regression Chart
    st.markdown("---")
    fig_reg = px.scatter(df_filtered, x="number_of_reviews", y="price", color="room_type", 
                         trendline="ols", title="Impact of Reviews on Price")
    st.plotly_chart(fig_reg, use_container_width=True)

# --- TAB 2: Market Segmentation (K-Means Clustering) ---
with tab2:
    st.header("Geographical Market Clusters")
    
    # Prepare data for clustering
    cluster_features = ['latitude', 'longitude', 'price']
    X_clust = df_filtered[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)
    
    # Run K-Means with k=3 as determined best
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualizing clusters on a map
    fig_map = px.scatter_mapbox(df_filtered, lat="latitude", lon="longitude", color="Cluster", 
                                size="price", hover_name="neighbourhood",
                                mapbox_style="carto-positron", zoom=10, height=600,
                                title="K-Means Market Segmentation (Location & Price)")
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3: Price Category (Classification) ---
with tab3:
    st.header("Classification: Premium vs Affordable")
    
    # Categorize data based on median price
    median_val = df['price'].median()
    df_filtered['Category'] = df_filtered['price'].apply(lambda x: "Premium" if x > median_val else "Affordable")
    
    # Distribution charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pie = px.pie(df_filtered, names='Category', title="Market Split", hole=0.4,
                         color_discrete_map={'Premium':'#005088', 'Affordable':'#11caa0'})
        st.plotly_chart(fig_pie)
    with col_c2:
        fig_bar = px.bar(df_filtered.groupby(['neighbourhood_group', 'Category']).size().reset_index(name='count'), 
                         x='neighbourhood_group', y='count', color='Category', barmode='group',
                         title="Inventory by Neighborhood and Category")
        st.plotly_chart(fig_bar)

# --- TAB 4: Anomaly Detection (Identifying Outliers) ---
with tab4:
    st.header("Anomalous Listings Detection")
    st.warning("Flagging unusual listings that deviate from typical market behavior.")
    
    # Run One-Class SVM on the filtered view
    X_ano = df_filtered[['price', 'number_of_reviews', 'availability_365']]
    scaler_ano = StandardScaler()
    X_scaled_ano = scaler_ano.fit_transform(X_ano)
    
    svm = OneClassSVM(nu=0.05, kernel="rbf") # Assume 5% anomalies for display
    df_filtered['Is_Anomaly'] = svm.fit_predict(X_scaled_ano)
    df_filtered['Status'] = df_filtered['Is_Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    # Scatter plot of anomalies
    fig_ano = px.scatter(df_filtered, x="price", y="number_of_reviews", color="Status",
                         color_discrete_map={'Normal':'#cbd5e1', 'Anomaly':'#ff4b4b'},
                         title="Price vs Reviews Anomaly Map")
    st.plotly_chart(fig_ano, use_container_width=True)

# --- 6. Business Insights Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Business Insights")
st.sidebar.success(f"""
**Key Observations:**
- **Diversity:** The market contains {df['room_type'].nunique()} room types across {df['neighbourhood_group'].nunique()} groups.
- **Hotspots:** Cluster analysis confirms Manhattan as the high-price tier.
- **Prediction:** Linear regression accuracy is optimized by encoding room types.
""")

# Option to view data
if st.sidebar.checkbox("Show Sample Data"):
    st.write(df_filtered.head(100))
