
# Import necessary libraries
import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify


%%writefile backend_files/app.py

# Import necessary libraries
import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify


# ---------------------------------------------------------
# Create the Flask application
# ---------------------------------------------------------
# Initialize the REST API that will serve predictions from
# the trained SuperKart machine learning model.
app = Flask("SuperKart Sales Predictor")


# ---------------------------------------------------------
# Load the serialized machine learning pipeline
# ---------------------------------------------------------
# Get the directory where app.py is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the complete path to the serialized model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_price_prediction_model_v1_0.joblib"
)

# Load the trained Random Forest prediction pipeline.
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Features expected by the trained model
# ---------------------------------------------------------
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Category"
]


# ---------------------------------------------------------
# Home-page endpoint
# ---------------------------------------------------------
@app.get("/")
def home():
    """
    Handles GET requests to the root URL.

    Returns a welcome message confirming that the
    SuperKart prediction API is running.
    """
    return "Welcome to the SuperKart Product Sales Prediction API!"


# ---------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    """
    Checks whether the API is running successfully.
    """
    return jsonify({
        "status": "healthy",
        "service": "SuperKart Sales Prediction API"
    })


# ---------------------------------------------------------
# Single-product prediction endpoint
# ---------------------------------------------------------
@app.post("/v1/sales")
def predict_sales():
    """
    Handles POST requests for one SuperKart product.

    The endpoint expects product and store information
    in JSON format and returns the predicted total sales.
    """

    # Read JSON data from the request body.
    product_data = request.get_json(silent=True)

    # Confirm that JSON data was supplied.
    if not product_data:
        return jsonify({
            "error": "No JSON data was provided."
        }), 400

    try:
        # Product_Category was created in the training notebook
        # from the first two characters of Product_Id.
        if "Product_Category" in product_data:
            product_category = product_data["Product_Category"]

        elif "Product_Id" in product_data:
            product_category = str(product_data["Product_Id"])[:2]

        else:
            return jsonify({
                "error": (
                    "Provide either Product_Id or Product_Category "
                    "in the request."
                )
            }), 400

        # Create one observation using the same feature names
        # used during model training.
        sample = {
            "Product_Weight": product_data["Product_Weight"],
            "Product_Sugar_Content": product_data[
                "Product_Sugar_Content"
            ],
            "Product_Allocated_Area": product_data[
                "Product_Allocated_Area"
            ],
            "Product_Type": product_data["Product_Type"],
            "Product_MRP": product_data["Product_MRP"],
            "Store_Id": product_data["Store_Id"],
            "Store_Establishment_Year": product_data[
                "Store_Establishment_Year"
            ],
            "Store_Size": product_data["Store_Size"],
            "Store_Location_City_Type": product_data[
                "Store_Location_City_Type"
            ],
            "Store_Type": product_data["Store_Type"],
            "Product_Category": product_category
        }

        # Convert the dictionary into a one-row DataFrame.
        input_data = pd.DataFrame([sample])

        # Ensure the columns appear in the same order
        # used during model training.
        input_data = input_data[MODEL_FEATURES]

        # Generate the sales prediction.
        predicted_sales = model.predict(input_data)[0]

        # Convert NumPy output into a regular Python float.
        predicted_sales = round(float(predicted_sales), 2)

        # Return the prediction as JSON.
        return jsonify({
            "Predicted Product Store Sales Total": predicted_sales
        })

    # Handle missing input features.
    except KeyError as error:
        return jsonify({
            "error": f"Missing required field: {error.args[0]}"
        }), 400

    # Handle missing input features.
    except Exception as error:
        return jsonify({
            "error": "The prediction could not be completed.",
            "details": str(error)
        }), 500


# ---------------------------------------------------------
# Batch-prediction endpoint
# ---------------------------------------------------------
@app.post("/v1/salesbatch")
def predict_sales_batch():
    """
    Handles batch predictions using an uploaded CSV file.

    The CSV must contain the same product and store features
    used when the machine learning model was trained.
    """

    # Confirm that a file was included in the request.
    if "file" not in request.files:
        return jsonify({
            "error": "No CSV file was uploaded."
        }), 400

    file = request.files["file"]

    # Confirm that the selected file has a filename.
    if file.filename == "":
        return jsonify({
            "error": "No CSV file was selected."
        }), 400

    try:
        # Read the uploaded CSV into a DataFrame.
        input_data = pd.read_csv(file)

        # Create Product_Category when the CSV contains Product_Id
        # but does not already contain Product_Category.
        if (
            "Product_Category" not in input_data.columns
            and "Product_Id" in input_data.columns
        ):
            input_data["Product_Category"] = (
                input_data["Product_Id"]
                .astype(str)
                .str[:2]
            )

        # Check for missing model features.
        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "The CSV is missing required columns.",
                "missing_columns": missing_features
            }), 400

        # Save product IDs for identifying the predictions.
        if "Product_Id" in input_data.columns:
            product_ids = input_data["Product_Id"].astype(str).tolist()
        else:
            product_ids = [
                f"Product_{index + 1}"
                for index in range(len(input_data))
            ]

        # Select only the columns expected by the model.
        model_input = input_data[MODEL_FEATURES]

        # Generate predictions.
        predicted_sales = model.predict(model_input)

        # Build the response.
        predictions = [
            {
                "Product_Id": product_id,
                "Predicted_Product_Store_Sales_Total": round(
                    float(prediction), 2
                )
            }
            for product_id, prediction
            in zip(product_ids, predicted_sales)
        ]

        return jsonify({
            "number_of_predictions": len(predictions),
            "predictions": predictions
        })

    except Exception as error:
        return jsonify({
            "error": "The batch prediction could not be completed.",
            "details": str(error)
        }), 500


# ---------------------------------------------------------
# Run the Flask application locally
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
app = Flask("SuperKart Sales Predictor")


# ---------------------------------------------------------
# Load the serialized machine learning pipeline
# ---------------------------------------------------------
# Get the directory where app.py is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the serialized model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_price_prediction_model_v1_0.joblib"
)

# Load the trained Random Forest pipeline.
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Features expected by the trained model
# ---------------------------------------------------------
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Category"
]


# ---------------------------------------------------------
# Home-page endpoint
# ---------------------------------------------------------
@app.get("/")
def home():
    """
    Handles GET requests to the root URL.

    Returns a welcome message confirming that the
    SuperKart prediction API is running.
    """
    return "Welcome to the SuperKart Product Sales Prediction API!"


# ---------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    """
    Checks whether the API is running successfully.
    """
    return jsonify({
        "status": "healthy",
        "service": "SuperKart Sales Prediction API"
    })


# ---------------------------------------------------------
# Single-product prediction endpoint
# ---------------------------------------------------------
@app.post("/v1/sales")
def predict_sales():
    """
    Handles POST requests for one SuperKart product.

    The endpoint expects product and store information
    in JSON format and returns the predicted total sales.
    """

    # Read JSON data from the request body.
    product_data = request.get_json(silent=True)

    # Confirm that JSON data was supplied.
    if not product_data:
        return jsonify({
            "error": "No JSON data was provided."
        }), 400

    try:
        # Product_Category was created in the training notebook
        # from the first two characters of Product_Id.
        if "Product_Category" in product_data:
            product_category = product_data["Product_Category"]

        elif "Product_Id" in product_data:
            product_category = str(product_data["Product_Id"])[:2]

        else:
            return jsonify({
                "error": (
                    "Provide either Product_Id or Product_Category "
                    "in the request."
                )
            }), 400

        # Create one observation using the same feature names
        # used during model training.
        sample = {
            "Product_Weight": product_data["Product_Weight"],
            "Product_Sugar_Content": product_data[
                "Product_Sugar_Content"
            ],
            "Product_Allocated_Area": product_data[
                "Product_Allocated_Area"
            ],
            "Product_Type": product_data["Product_Type"],
            "Product_MRP": product_data["Product_MRP"],
            "Store_Id": product_data["Store_Id"],
            "Store_Establishment_Year": product_data[
                "Store_Establishment_Year"
            ],
            "Store_Size": product_data["Store_Size"],
            "Store_Location_City_Type": product_data[
                "Store_Location_City_Type"
            ],
            "Store_Type": product_data["Store_Type"],
            "Product_Category": product_category
        }

        # Convert the dictionary into a one-row DataFrame.
        input_data = pd.DataFrame([sample])

        # Ensure the columns appear in the same order
        # used during model training.
        input_data = input_data[MODEL_FEATURES]

        # Generate the sales prediction.
        predicted_sales = model.predict(input_data)[0]

        # Convert NumPy output into a regular Python float.
        predicted_sales = round(float(predicted_sales), 2)

        # Return the prediction as JSON.
        return jsonify({
            "Predicted Product Store Sales Total": predicted_sales
        })

    except KeyError as error:
        return jsonify({
            "error": f"Missing required field: {error.args[0]}"
        }), 400

    except Exception as error:
        return jsonify({
            "error": "The prediction could not be completed.",
            "details": str(error)
        }), 500


# ---------------------------------------------------------
# Batch-prediction endpoint
# ---------------------------------------------------------
@app.post("/v1/salesbatch")
def predict_sales_batch():
    """
    Handles batch predictions using an uploaded CSV file.

    The CSV must contain the same product and store features
    used when the machine learning model was trained.
    """

    # Confirm that a file was included in the request.
    if "file" not in request.files:
        return jsonify({
            "error": "No CSV file was uploaded."
        }), 400

    file = request.files["file"]

    # Confirm that the selected file has a filename.
    if file.filename == "":
        return jsonify({
            "error": "No CSV file was selected."
        }), 400

    try:
        # Read the uploaded CSV into a DataFrame.
        input_data = pd.read_csv(file)

        # Create Product_Category when the CSV contains Product_Id
        # but does not already contain Product_Category.
        if (
            "Product_Category" not in input_data.columns
            and "Product_Id" in input_data.columns
        ):
            input_data["Product_Category"] = (
                input_data["Product_Id"]
                .astype(str)
                .str[:2]
            )

        # Check for missing model features.
        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "The CSV is missing required columns.",
                "missing_columns": missing_features
            }), 400

        # Save product IDs for identifying the predictions.
        if "Product_Id" in input_data.columns:
            product_ids = input_data["Product_Id"].astype(str).tolist()
        else:
            product_ids = [
                f"Product_{index + 1}"
                for index in range(len(input_data))
            ]

        # Select only the columns expected by the model.
        model_input = input_data[MODEL_FEATURES]

        # Generate predictions.
        predicted_sales = model.predict(model_input)

        # Build the response.
        predictions = [
            {
                "Product_Id": product_id,
                "Predicted_Product_Store_Sales_Total": round(
                    float(prediction), 2
                )
            }
            for product_id, prediction
            in zip(product_ids, predicted_sales)
        ]

        return jsonify({
            "number_of_predictions": len(predictions),
            "predictions": predictions
        })

    except Exception as error:
        return jsonify({
            "error": "The batch prediction could not be completed.",
            "details": str(error)
        }), 500


# ---------------------------------------------------------
# Run the Flask application locally
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
