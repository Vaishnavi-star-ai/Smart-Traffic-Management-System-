import streamlit as st
import pandas as pd
import pymysql
import plotly.express as px

# --------------------------
# Database Connection
# --------------------------
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='traffic_signal'
    )

# --------------------------
# Fetch Data from MySQL
# --------------------------
def fetch_traffic_data():
    conn = get_db_connection()
    query = "SELECT * FROM traffic_log ORDER BY date_time DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --------------------------
# Streamlit App UI
# --------------------------
def main():
    st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
    st.title("🚦 Smart Traffic Signal Dashboard")
    st.markdown("Visualizing traffic signal performance in real-time from MySQL")

    # Load traffic data
    df = fetch_traffic_data()

    if df.empty:
        st.warning("No traffic data found in the database.")
        return

    # ----------------------------------
    # Data Table
    # ----------------------------------
    st.subheader("📋 Traffic Log Data")
    st.dataframe(df, use_container_width=True)

    # ----------------------------------
    # Latest Cycle Visualizations
    # ----------------------------------
    latest_cycle = df['cycle'].max()
    latest_data = df[df['cycle'] == latest_cycle]

    st.markdown(f"### 🚗 Traffic Stats for Cycle {latest_cycle}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Vehicle Count")
        fig1 = px.bar(
            latest_data,
            x='direction',
            y='total_vehicles',
            color='direction',
            labels={'total_vehicles': 'Total Vehicles'},
            title=f"Total Vehicles per Direction (Cycle {latest_cycle})"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Green Light Time")
        fig2 = px.bar(
            latest_data,
            x='direction',
            y='green_time',
            color='direction',
            labels={'green_time': 'Green Time (s)'},
            title=f"Green Time per Direction (Cycle {latest_cycle})"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------
    # Trend Analysis Over Time
    # ----------------------------------
    st.markdown("### 📈 Trends Over Multiple Cycles")

    cycle_agg = df.groupby('cycle').agg({
        'total_vehicles': 'sum',
        'green_time': 'mean'
    }).reset_index()

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.line(
            cycle_agg,
            x='cycle',
            y='total_vehicles',
            title="Total Vehicles Over Cycles",
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.line(
            cycle_agg,
            x='cycle',
            y='green_time',
            title="Average Green Time Over Cycles",
            markers=True
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Developed by Rohith | Smart Traffic Management System")

# Run the app
if __name__ == "__main__":
    main()
