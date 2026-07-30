
import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# Backend API configuration
# ---------------------------------------------------------
# "backend" is the Flask container name used on the
# shared Docker network.
BACKEND_URL = "http://backend:7860"


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("SuperKart Product Sales Prediction")

st.write(
    "Enter the product and store information below to predict "
    "the total sales for a SuperKart product."
)


# ---------------------------------------------------------
# Online prediction section
# ---------------------------------------------------------
st.subheader("Online Prediction")


# Product information
product_id = st.text_input(
    "Product ID",
    value="FDX07",
    help=(
        "The first two characters of the Product ID are used "
        "to create Product Category."
    )
)

product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=10.0,
    step=0.1
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "No Sugar",
        "Regular",
        "reg"
    ]
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.05,
    step=0.01,
    format="%.3f"
)

product_type = st.selectbox(
    "Product Type",
    [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=150.0,
    step=1.0
)


# Store information
store_id = st.selectbox(
    "Store ID",
    [
        "OUT001",
        "OUT002",
        "OUT003",
        "OUT004"
    ]
)

store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1900,
    max_value=2100,
    value=1999,
    step=1
)

store_size = st.selectbox(
    "Store Size",
    [
        "High",
        "Medium",
        "Small"
    ]
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)


# ---------------------------------------------------------
# Build the JSON request for one prediction
# ---------------------------------------------------------
input_data = {
    "Product_Id": product_id,
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}


# ---------------------------------------------------------
# Submit one prediction
# ---------------------------------------------------------
if st.button("Predict Sales", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/sales",
            json=input_data,
            timeout=30
        )

        if response.status_code == 200:
            prediction = response.json()[
                "Predicted Product Store Sales Total"
            ]

            st.success(
                "Predicted Product Store Sales Total: "
                f"${prediction:,.2f}"
            )

        else:
            error_response = response.json()

            st.error(
                error_response.get(
                    "error",
                    "The prediction request was unsuccessful."
                )
            )

            if "details" in error_response:
                st.write(error_response["details"])

    except requests.exceptions.RequestException as error:
        st.error(
            "Unable to connect to the SuperKart prediction API. "
            f"Details: {error}"
        )


# ---------------------------------------------------------
# Batch prediction section
# ---------------------------------------------------------
st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing multiple products and stores."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file for batch prediction",
    type=["csv"]
)


# ---------------------------------------------------------
# Submit batch predictions
# ---------------------------------------------------------
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/salesbatch",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                st.success(
                    f"Batch prediction completed for "
                    f"{len(predictions)} products."
                )

                predictions_df = pd.DataFrame(predictions)

                st.dataframe(
                    predictions_df,
                    use_container_width=True
                )

                csv_output = predictions_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions",
                    data=csv_output,
                    file_name="superkart_predictions.csv",
                    mime="text/csv"
                )

            else:
                error_response = response.json()

                st.error(
                    error_response.get(
                        "error",
                        "The batch prediction request was unsuccessful."
                    )
                )

                if "missing_columns" in error_response:
                    st.write(
                        "Missing columns:",
                        error_response["missing_columns"]
                    )

                if "details" in error_response:
                    st.write(error_response["details"])

        except requests.exceptions.RequestException as error:
            st.error(
                "Unable to connect to the SuperKart prediction API. "
                f"Details: {error}"
            )
