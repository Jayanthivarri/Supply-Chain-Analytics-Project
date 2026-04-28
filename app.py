# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# PAGE CONFIG 
st.set_page_config(
    page_title="Delivery Performance, Delay Risk, & Logistics Efficiency Analysis in Global Supply Chain Operations",
    layout="wide",
    page_icon="📦"
)

df=pd.read_csv("APL_Logistics.csv",encoding="latin1")
model=joblib.load("late_delivery_model.pkl")

# create new columns
df["delay_gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

def status(x):
    if x == 0:
        return "On-time"
    elif x > 0:
        return "Delayed"
    else:
        return "Early"

df["delivery_result"] = df["delay_gap"].apply(status)

#  SIDEBAR
st.sidebar.title("📌 Logistics Configuration")

shipping_mode = st.sidebar.multiselect("Select Shipping Mode",
    df["Shipping Mode"].unique(),
    default=df["Shipping Mode"].unique()
)

region = st.sidebar.multiselect(
    "Select Region",
    df["Order Region"].unique(),
    default=df["Order Region"].unique()
)

market = st.sidebar.multiselect(
    "Select Market",
    df["Market"].unique(),
    default=df["Market"].unique()
)

segment = st.sidebar.multiselect(
    "Select Customer Segment",
    df["Customer Segment"].unique(),
    default=df["Customer Segment"].unique()
)

module = st.sidebar.radio(
    "Select Module",
    [
        "📊 Overview",
        "⚠ Delay Risk",
        "🚚 Shipping Mode",
        "🌍 Regional Analysis",
        "🤖 Prediction",
        "🧠 Model Insights"
    ]
)

# FILTER DATA 
filtered_df = df[
    (df["Shipping Mode"].isin(shipping_mode)) &
    (df["Order Region"].isin(region)) &
    (df["Market"].isin(market)) &
    (df["Customer Segment"].isin(segment))
]

# KPI CALCULATIONS 
on_time = round((filtered_df["delivery_result"] == "On-time").mean() * 100, 2)
sla = round((filtered_df["delivery_result"] != "Delayed").mean() * 100, 2)
avg_delay = round(filtered_df["delay_gap"].mean(), 2)
late_risk = round(filtered_df["Late_delivery_risk"].mean() * 100, 2)

mode_eff = filtered_df.groupby("Shipping Mode").apply(lambda x:(x["delivery_result"]!="Delayed").mean()*100)
shipping_eff=round(mode_eff.max(),2)
region_delay = filtered_df.groupby("Order Region").apply(lambda x:(x["delivery_result"]=="Delayed").mean()*100)
regional_index=round(region_delay.max(),2)

#  TITLE 
col1, col2, col3 = st.columns([1.2,4,1.2])

with col1:
    st.image("logo_apl.png", width=200)

with col2:
    st.title("📦 Delivery Performance, Delay Risk & Logistics Efficiency Analysis")
    st.caption("Interactive dashboard for logistics efficiency, delay diagnostics, and Global Supply Chain Performance.")

with col3:
    st.image("Logo_unified.png", width=110)


#  KPI CARDS 
st.subheader("📊 Key Performance Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("On-Time %", f"{on_time}%")
c2.metric("SLA Success %", f"{sla}%")
c3.metric("Avg Delay", f"{avg_delay}days")
c4.metric("Late Risk %", f"{late_risk}%")
c5.metric("Shipping Efficiency %", f"{shipping_eff}%")
c6.metric("Regional Delay Index", f"{regional_index}%")

st.markdown("---")


# OVERVIEW 
if module == "📊 Overview":
    st.subheader("📊 Delivery Performance Overview")
    
    st.markdown("___")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Delivery Result Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            x="delivery_result",
            data=filtered_df,
            order=["Early", "On-time", "Delayed"],
            color="orange",
            ax=ax
        )
        ax.set_xlabel("Delivery Status")
        ax.set_ylabel("Orders Count")
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Average Delay Scorecards")
        mode_delay = (
            filtered_df.groupby("Shipping Mode")["delay_gap"]
            .mean()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(6,4))

        mode_delay.plot(
            kind="bar",
            color="orange",
            #width=0.6,
            ax=ax
        )

        ax.set_xlabel("Shipping Mode")
        ax.set_ylabel("Avg Delay (Days)")
        ax.tick_params(axis='x',labelsize=10,rotation=20)
        ax.tick_params(axis='y', labelsize=10)
        st.pyplot(fig)

#  DELAY RISK 
elif module == "⚠ Delay Risk":
    st.subheader("⚠️ Delay Risk Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Late Delivery Risk Distribution")
        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Late_delivery_risk",
            data=filtered_df,
            color="orange",
            ax=ax
        )

        ax.set_xlabel("Late Delivery Risk")
        ax.set_ylabel("Orders Count")
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        st.pyplot(fig)

    with col2:
        st.subheader("Delay Gap Histogram")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            data=filtered_df,
            x="delay_gap",
            bins=[-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5],
            kde=True,
            color="orange",
            edgecolor="black",
            ax=ax
        )

        ax.set_xlabel("Delay Gap (Days)")
        ax.set_ylabel("Orders Count")
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        st.pyplot(fig)

    
# SHIPPING MODE 
elif module == "🚚 Shipping Mode":

    st.subheader("🚚 Shipping Mode Comparison Dashboard")

    #st.subheader("Shipping Mode vs Delay Gap")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Mode-wise Delay Performance")
        mode_delay = (
            filtered_df.groupby("Shipping Mode")["delay_gap"].mean().sort_values())

        fig, ax = plt.subplots(figsize=(6,4))
        mode_delay.plot(
            kind="bar",
            color="green",
            width=0.6,
            ax=ax
        )

        ax.set_xlabel("Shipping Mode")
        ax.set_ylabel("Avg Delay")
        ax.tick_params(axis='x', rotation=20)

        st.pyplot(fig)
    with col2:
        st.subheader("SLA Compliance by Mode")

        sla_mode = (
            filtered_df.groupby("Shipping Mode")["delay_gap"].apply(lambda x: (x <= 0).mean() * 100).sort_values()
        )

        fig, ax = plt.subplots(figsize=(6,4.2))

        sla_mode.plot(
            kind="bar",
            color="green",
            width=0.6,
            ax=ax
        )

        ax.set_xlabel("Shipping Mode")
        ax.set_ylabel("SLA Success %")
        ax.tick_params(axis='x', rotation=20)

        plt.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Shipping Mode vs Delivery Result")

        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            x="Shipping Mode",
            hue="delivery_result",
            data=filtered_df,
            ax=ax
        )
        ax.tick_params(axis='x', rotation=25)
        st.pyplot(fig)


    with col4:
        st.markdown("### Delay Distribution by Shipping Mode")

        fig, ax = plt.subplots(figsize=(6.5,4))
        sns.boxplot(
            x="Shipping Mode",
            y="delay_gap",
            data=filtered_df,
            color="green",
            showfliers=False,
            ax=ax
        )
        ax.tick_params(axis='x', rotation=25)
        st.pyplot(fig)


#  REGIONAL analysis
elif module == "🌍 Regional Analysis":
    st.subheader("🌍 Regional & Market Diagnostics")
    st.subheader("Average Delay Gap by Region")
    #col1, col2 = st.columns(2)
    left,center,right=st.columns([1,4,1])
  
    with center:
        
        region_delay = filtered_df.groupby("Order Region")["delay_gap"].mean()
        region_delay = region_delay.sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10,5))

        region_delay.sort_values().plot(
            kind="bar",
            color="skyblue",
            ax=ax
        )
        #ax.set_title("Average Delay Gap by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Avg Delay")
        #ax.set_xlim(0.35, 0.66)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
    st.markdown("___")

    st.subheader("Market-wise Logistics Efficiency")
    left,center,right=st.columns([1,4,1])
   
    with center:

        market_eff = (
            filtered_df.groupby("Market")["delay_gap"]\
            .apply(lambda x: (x <= 0).mean() * 100)\
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10,5))

        market_eff.plot(
            kind="bar",
            color="green",
            width=0.7,
            ax=ax
        )

        #ax.set_xlabel("Market")
        ax.set_ylabel("Efficiency %")
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(40, 45)

        st.pyplot(fig)


#  PREDICTION 
elif module == "🤖 Prediction":
    st.subheader("🤖 Late Delivery Prediction Dashboard")

    st.subheader("Late Delivery Risk Prediction")

    #st.info("Load your saved model file: late_delivery_model.pkl")

    try:
        model = joblib.load("late_delivery_model.pkl")

        col1, col2 = st.columns(2)

        with col1:
            p_mode = st.selectbox("Shipping Mode", df["Shipping Mode"].unique())
            p_region = st.selectbox("Region", df["Order Region"].unique())
            p_market = st.selectbox("Market", df["Market"].unique())
            p_segment = st.selectbox("Customer Segment", df["Customer Segment"].unique())
            p_type = st.selectbox("Order Type", df["Type"].unique())

        with col2:
            p_sched = st.number_input("Scheduled Shipping Days", 0, 10, 4)
            p_qty = st.number_input("Order Quantity", 1, 50, 2)
            p_sales = st.number_input("Sales", 0.0, 10000.0, 500.0)
            p_profit = st.number_input("Benefit per Order", -500.0, 5000.0, 100.0)
            p_disc = st.number_input("Order Item Discount", 0.0, 1.0, 0.1)
            p_price = st.number_input("Product Price", 0.0, 5000.0, 100.0)

        if st.button("Predict Risk"):

            input_df = pd.DataFrame({
                "Shipping Mode": [p_mode],
                "Order Region": [p_region],
                "Market": [p_market],
                "Customer Segment": [p_segment],
                "Type": [p_type],
                "Days for shipment (scheduled)": [p_sched],
                "Order Item Quantity": [p_qty],
                "Sales": [p_sales],
                "Benefit per order": [p_profit],
                "Order Item Discount": [p_disc],
                "Order Item Product Price": [p_price]
            })

            pred = model.predict(input_df)[0]

            if pred == 1:
                st.error("⚠ High Late Delivery Risk")
            else:
                st.success("✅ Low Delivery Risk")

    except:
        st.warning("Model file not found.")



elif module == "🧠 Model Insights":

    st.subheader("🧠 Model Insights Dashboard")

    try:
        model = joblib.load("late_delivery_model.pkl")

        # same training features
        features = [
            "Shipping Mode",
            "Order Region",
            "Market",
            "Customer Segment",
            "Type",
            "Days for shipment (scheduled)",
            "Order Item Quantity",
            "Sales",
            "Benefit per order",
            "Order Item Discount",
            "Order Item Product Price"
        ]

        X = df[features]
        y = df["Late_delivery_risk"]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_test)

        # Live Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Accuracy", f"{acc*100:.2f}%")
        with c2:
            st.metric("Precision", f"{prec*100:.2f}%")
        with c3:
            st.metric("Recall", f"{rec*100:.2f}%")
        with c4:
            st.metric("F1 Score", f"{f1*100:.2f}%")

        st.markdown("---")

        #col1, col2 = st.columns(2)

        # Confusion Matrix
        
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        left,center,right=st.columns([1,2,1])
        with center:
            fig, ax = plt.subplots(figsize=(5,4))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["No Delay", "Delay"],
                yticklabels=["No Delay", "Delay"],
                linewidths=1,
                linecolor="white",
                ax=ax
            )

            ax.set_xlabel("Predicted",fontsize=11)
            ax.set_ylabel("Actual",fontsize=11)

            st.pyplot(fig)
        st.markdown("___")

        # Feature Importance
        st.subheader("Feature Importance")

        rf = model.named_steps["rf"]
        prep = model.named_steps["prep"]

        feature_names = prep.get_feature_names_out()
        importance = rf.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
        
        left, center, right = st.columns([1,2,1])
        with center:
            fig, ax = plt.subplots(figsize=(6,4))

            ax.barh(
                imp_df["Feature"][::-1],
                imp_df["Importance"][::-1],
                color="green"
                )

            ax.set_xlabel("Importance Score",fontsize=11)
            ax.set_ylabel("Features",fontsize=11)

            st.pyplot(fig)


        st.markdown("---")

        st.subheader("📌 Key Insights")

        st.markdown("""
        - Second Class shipments show highest delay probability  
        - Central Asia region has highest average delay  
        - Africa market shows strongest logistics efficiency  
        - Shipping Mode is top predictor of late delivery  
        """)


        st.subheader("💼 Business Recommendations")

        st.markdown("""
        - Improve SLA monitoring for Second Class deliveries  
        - Allocate extra resources to high-risk regions  
        - Use ML alerts for predicted delayed shipments  
        - Optimize carrier planning by market performance  
        """)
    
    except:
        st.warning("Model file not found or incompatible.")

st.markdown("---")
st.write("👨‍💻 Developed By Jayanthi Varri")
st.write("🔗 LinkedIn: https://www.linkedin.com/in/jayanthi-varri/")
