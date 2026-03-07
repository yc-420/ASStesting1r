import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "garments_worker_productivity.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.joblib")
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
LIN_MODEL_PATH = os.path.join(BASE_DIR, "lin_model.joblib")
RIDGE_MODEL_PATH = os.path.join(BASE_DIR, "ridge_model.joblib")
DT_MODEL_PATH = os.path.join(BASE_DIR, "dt_model.joblib")

st.set_page_config(page_title="Garment Worker Productivity Dashboard", layout="wide")


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df["department"] = df["department"].astype(str).str.strip().str.lower().replace({"sweing": "sewing"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.day_name()
    df["wip"] = df["wip"].fillna(0)
    return df


@st.cache_data
def build_model_dataframe():
    df = load_raw_data().copy()
    model_df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)
    model_df = model_df.drop(columns=["date"])
    model_df.columns = model_df.columns.str.strip()
    return model_df


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


@st.cache_resource
def train_and_evaluate_models():
    model_df = build_model_dataframe()
    X = model_df.drop("actual_productivity", axis=1)
    y = model_df["actual_productivity"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    predictions = {}
    best_models = {}

    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(Xtrain, ytrain)
    pred_base = baseline.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_base)
    cv_rmse = -cross_val_score(baseline, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error").mean()
    cv_r2 = cross_val_score(baseline, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Baseline",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": "Mean strategy"
    })
    predictions["Baseline"] = pred_base
    best_models["Baseline"] = baseline

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(Xtrain, ytrain)
    pred_lin = linreg.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_lin)
    cv_rmse = -cross_val_score(linreg, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error").mean()
    cv_r2 = cross_val_score(linreg, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": "Default"
    })
    predictions["Linear Regression"] = pred_lin
    best_models["Linear Regression"] = linreg

    # Ridge Regression
    ridge_grid = GridSearchCV(
        Ridge(),
        {"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    ridge_grid.fit(Xtrain, ytrain)
    best_ridge = ridge_grid.best_estimator_
    pred_ridge = best_ridge.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_ridge)
    cv_rmse = -cross_val_score(best_ridge, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error").mean()
    cv_r2 = cross_val_score(best_ridge, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Ridge Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": str(ridge_grid.best_params_)
    })
    predictions["Ridge Regression"] = pred_ridge
    best_models["Ridge Regression"] = best_ridge

    # Decision Tree
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    dt_grid.fit(Xtrain, ytrain)
    best_dt = dt_grid.best_estimator_
    pred_dt = best_dt.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_dt)
    cv_rmse = -cross_val_score(best_dt, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error").mean()
    cv_r2 = cross_val_score(best_dt, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Decision Tree",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": str(dt_grid.best_params_)
    })
    predictions["Decision Tree"] = pred_dt
    best_models["Decision Tree"] = best_dt

    # Random Forest (load saved best model if exists, else tune)
    if os.path.exists(RF_MODEL_PATH):
        try:
            best_rf = joblib.load(RF_MODEL_PATH)
        except Exception:
            best_rf = None
    else:
        best_rf = None

    if best_rf is None:
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        rf_grid.fit(Xtrain, ytrain)
        best_rf = rf_grid.best_estimator_
        rf_best_params = str(rf_grid.best_params_)
    else:
        rf_best_params = getattr(best_rf, "get_params", lambda: {})()
        rf_best_params = str({k: rf_best_params[k] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"] if k in rf_best_params})
        best_rf.fit(Xtrain, ytrain)

    pred_rf = best_rf.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_rf)
    cv_rmse = -cross_val_score(best_rf, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error").mean()
    cv_r2 = cross_val_score(best_rf, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": rf_best_params
    })
    predictions["Random Forest"] = pred_rf
    best_models["Random Forest"] = best_rf

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return {
        "Xtrain": Xtrain,
        "Xtest": Xtest,
        "ytrain": ytrain,
        "ytest": ytest,
        "results_df": results_df,
        "predictions": predictions,
        "best_models": best_models,
        "feature_columns": list(X.columns),
    }


@st.cache_data
def get_column_details():
    return pd.DataFrame([
        ["date", "Production date", "datetime"],
        ["quarter", "Production quarter", "categorical"],
        ["department", "Department name", "categorical"],
        ["team", "Team number", "numeric"],
        ["targeted_productivity", "Target productivity rate", "numeric"],
        ["smv", "Standard Minute Value", "numeric"],
        ["wip", "Work in progress", "numeric"],
        ["over_time", "Overtime minutes", "numeric"],
        ["incentive", "Incentive amount", "numeric"],
        ["idle_time", "Idle time", "numeric"],
        ["idle_men", "Number of idle workers", "numeric"],
        ["no_of_style_change", "Count of style changes", "numeric"],
        ["no_of_workers", "Number of workers", "numeric"],
        ["actual_productivity", "Actual productivity achieved", "target"],
        ["day", "Day name derived from date", "derived categorical"],
    ], columns=["Feature", "Description", "Type"])


raw_df = load_raw_data()
model_bundle = train_and_evaluate_models()
results_df = model_bundle["results_df"]
feature_cols = model_bundle["feature_columns"]
best_models = model_bundle["best_models"]

st.title("Garment Worker Productivity Dashboard")
st.caption("BMDS2003 Data Science Project - full prototype with EDA, model performance, prediction, and batch prediction")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Exploration", "Model Performance", "Single Prediction", "Batch Prediction", "About"],
)

if menu == "Overview":
    st.header("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing WIP", int(raw_df["wip"].isna().sum()))
    c4.metric("Teams", int(raw_df["team"].nunique()))

    st.subheader("Business Objective")
    st.write(
        "This project predicts garment worker actual productivity and compares multiple regression models "
        "to support productivity planning, workforce management, and operational decision-making."
    )

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Column Details")
    st.dataframe(get_column_details(), use_container_width=True)

    st.subheader("Data Preparation Summary")
    st.markdown(
        "- Missing values in **wip** were filled with 0.\n"
        "- Department value **sweing** was corrected to **sewing**.\n"
        "- **date** was converted to datetime and **day** was derived from date.\n"
        "- Categorical features were encoded using one-hot encoding for model training."
    best_model_row = results_df.sort_values("RMSE").iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset Size", len(raw_df))
col2.metric("Features", raw_df.shape[1])
col3.metric("Best Model", best_model_row["Model"])
col4.metric("Best R²", f"{best_model_row['R2']:.4f}")
    )

elif menu == "Data Exploration":
    st.header("Data Exploration")

    # fresh copy for filtering/display
    eda_df = raw_df.copy()
    eda_df["department"] = eda_df["department"].astype(str).str.strip().str.lower()
    eda_df["department"] = eda_df["department"].replace({"sweing": "sewing"})

    # filters
    st.subheader("Filters")
    colf1, colf2 = st.columns(2)

    with colf1:
        dept_options = ["All"] + sorted(eda_df["department"].dropna().unique().tolist())
        dept_filter = st.selectbox("Select Department", dept_options)

    with colf2:
        quarter_options = ["All"] + sorted(eda_df["quarter"].dropna().unique().tolist())
        quarter_filter = st.selectbox("Select Quarter", quarter_options)

    filtered_df = eda_df.copy()

    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]

    if quarter_filter != "All":
        filtered_df = filtered_df[filtered_df["quarter"] == quarter_filter]

    st.write(f"Showing {len(filtered_df)} records")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df["actual_productivity"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Actual Productivity")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=filtered_df["actual_productivity"], ax=ax)
    ax.set_title("Boxplot of Actual Productivity")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="targeted_productivity",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Targeted vs Actual Productivity")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="over_time",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Over Time vs Actual Productivity")
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="no_of_workers",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Workers vs Actual Productivity")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="incentive",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Incentive vs Actual Productivity")
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(
            x="department",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Actual Productivity by Department")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(
            x="quarter",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Actual Productivity by Quarter")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(
        x="day",
        y="actual_productivity",
        data=filtered_df,
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"],
        ax=ax,
    )
    ax.set_title("Actual Productivity by Day")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    team_avg = filtered_df.groupby("team")["actual_productivity"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    team_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Actual Productivity by Team")
    ax.set_xlabel("Team")
    ax.set_ylabel("Average Actual Productivity")
    st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="smv",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("SMV vs Actual Productivity")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(
            x="idle_time",
            y="actual_productivity",
            data=filtered_df,
            ax=ax
        )
        ax.set_title("Idle Time vs Actual Productivity")
        st.pyplot(fig)

elif menu == "Model Performance":
    st.header("Model Performance")

    best_model = results_df.sort_values("RMSE").iloc[0]

    st.success(
        f"Best Performing Model: {best_model['Model']} "
        f"(RMSE = {best_model['RMSE']:.4f}, R² = {best_model['R2']:.4f})"
    )

    st.info(
        "The best model is selected based on the lowest RMSE and stronger R² performance."
    )

    display_df = results_df.copy()
    for col in ["MAE", "RMSE", "R2", "CV_RMSE", "CV_R2"]:
        display_df[col] = display_df[col].round(4)

    st.dataframe(display_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax)
    ax.set_title("RMSE Comparison Across Models")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results_df, x="Model", y="R2", ax=ax)
    ax.set_title("R² Comparison Across Models")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    rf_model = best_models["Random Forest"]
    if hasattr(rf_model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": model_bundle["Xtrain"].columns,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 15 Feature Importances (Random Forest)")
        st.pyplot(fig)

    selected_model = st.selectbox(
        "Choose a model for Actual vs Predicted plot",
        list(model_bundle["predictions"].keys()),
        index=4
    )

    ytest = model_bundle["ytest"]
    ypred = model_bundle["predictions"][selected_model]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ytest, ypred, alpha=0.7)
    min_v = min(float(np.min(ytest)), float(np.min(ypred)))
    max_v = max(float(np.max(ytest)), float(np.max(ypred)))
    ax.plot([min_v, max_v], [min_v, max_v], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted - {selected_model}")
    st.pyplot(fig)

elif menu == "Single Prediction":
    st.header("Single Prediction")
    st.write("Enter production information to predict actual productivity.")

    col1, col2, col3 = st.columns(3)
    with col1:
        team = st.number_input("Team", min_value=1, max_value=50, value=8)
        targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.80, 0.01)
        smv = st.number_input("SMV", min_value=0.0, value=26.16)
        wip = st.number_input("WIP", min_value=0.0, value=1108.0)
    with col2:
        over_time = st.number_input("Over Time", min_value=0.0, value=7080.0)
        incentive = st.number_input("Incentive", min_value=0.0, value=98.0)
        idle_time = st.number_input("Idle Time", min_value=0.0, value=0.0)
        idle_men = st.number_input("Idle Men", min_value=0.0, value=0.0)
    with col3:
        no_of_style_change = st.number_input("No. of Style Change", min_value=0, value=0)
        no_of_workers = st.number_input("No. of Workers", min_value=1.0, value=59.0)
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        department = st.selectbox("Department", ["finishing", "sewing"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    model_choice = st.selectbox(
        "Prediction Model",
        ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"],
        index=3
    )

    raw = {
        "team": team,
        "targeted_productivity": targeted_productivity,
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_style_change": no_of_style_change,
        "no_of_workers": no_of_workers,
        "quarter": quarter,
        "department": department,
        "day": day,
    }

    pred_df = pd.DataFrame([raw])
    pred_df = pd.get_dummies(pred_df, columns=["quarter", "department", "day"], drop_first=True)

    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = 0

    pred_df = pred_df[feature_cols].replace({True: 1, False: 0})

    if st.button("Predict"):
        model = best_models[model_choice]
        pred = float(model.predict(pred_df)[0])

        st.metric("Predicted actual_productivity", f"{pred:.4f}")
        st.dataframe(pd.DataFrame([raw]), use_container_width=True)

        target = targeted_productivity
        gap = pred - target

        st.write("### Productivity Comparison")
        st.write(f"Target Productivity: {target:.3f} ({target*100:.2f}%)")
        st.write(f"Predicted Productivity: {pred:.3f} ({pred*100:.2f}%)")
        st.write(f"Performance Gap: {gap:.3f}")

        if pred >= target:
            st.success("Status: On Track / Overachievement")
            st.info("The team is likely to achieve or exceed the targeted productivity.")
        else:
            st.warning("Status: Under Target")
            st.info("The team may not reach the targeted productivity under the current production conditions.")
        
elif menu == "Batch Prediction":
    st.header("Batch Prediction")
    st.write(
        "Upload a CSV file containing multiple garment production records to generate productivity predictions."
    )

    st.subheader("Sample Input Format")

    template_df = pd.DataFrame([{
        "team": 8,
        "targeted_productivity": 0.80,
        "smv": 26.16,
        "wip": 1108,
        "over_time": 7080,
        "incentive": 98,
        "idle_time": 0,
        "idle_men": 0,
        "no_of_style_change": 0,
        "no_of_workers": 59,
        "quarter": "Quarter1",
        "department": "sewing",
        "day": "Monday"
    }])

    st.dataframe(template_df, use_container_width=True)

    st.download_button(
        "Download Sample Template",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_choice = st.selectbox(
        "Model for Batch Prediction",
        ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"],
        index=3
    )

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        batch_df.columns = batch_df.columns.str.strip()

        if "day" not in batch_df.columns and "date" in batch_df.columns:
            batch_df["date"] = pd.to_datetime(batch_df["date"], errors="coerce")
            batch_df["day"] = batch_df["date"].dt.day_name()

        if "department" in batch_df.columns:
            batch_df["department"] = (
                batch_df["department"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"sweing": "sewing"})
            )

        if "wip" in batch_df.columns:
            batch_df["wip"] = batch_df["wip"].fillna(0)

        keep_cols = [
            "team", "targeted_productivity", "smv", "wip", "over_time", "incentive",
            "idle_time", "idle_men", "no_of_style_change", "no_of_workers",
            "quarter", "department", "day"
        ]

        missing_cols = [c for c in keep_cols if c not in batch_df.columns]

        if missing_cols:
            st.error("The uploaded file format does not match the required structure.")
            st.info("Please use the sample template above.")
        else:
            pred_input = batch_df[keep_cols].copy()

            pred_input = pd.get_dummies(
                pred_input,
                columns=["quarter", "department", "day"],
                drop_first=True
            )

            for c in feature_cols:
                if c not in pred_input.columns:
                    pred_input[c] = 0

            pred_input = pred_input[feature_cols].replace({True: 1, False: 0})

            model = best_models[model_choice]

            batch_df["predicted_actual_productivity"] = model.predict(pred_input)

            st.subheader("Prediction Results")
            st.dataframe(batch_df.head(20), use_container_width=True)

            st.download_button(
                "Download Results CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_prediction_results.csv",
                mime="text/csv",
            )
elif menu == "About":
    st.header("About This Project")

    st.markdown("""
    ### Garment Worker Productivity Dashboard

    This interactive dashboard was developed as part of the **BMDS2003 Data Science group assignment** at  
    **Tunku Abdul Rahman University of Management and Technology (TARUMT)**.

    The goal of this project is to apply **data science and machine learning techniques** to analyse and predict  
    **garment worker productivity** using operational production data.

    ### Project Objective
    The main objective is to predict **actual productivity** of garment worker teams based on production conditions such as:
    - team size
    - overtime
    - incentives
    - work-in-progress (WIP)
    - department and production quarter

    By predicting productivity levels, production managers can evaluate whether current working conditions are sufficient to meet productivity targets.

    ### Dataset
    The project uses the **Garment Worker Productivity dataset**, which contains production records of garment factory teams including:
    - production targets
    - workforce size
    - department information
    - overtime and incentive data
    - actual productivity levels

    ### Data Science Methods Used
    The following techniques were applied in this project:

    - Data cleaning and preprocessing
    - Exploratory Data Analysis (EDA)
    - Baseline model using Dummy Regressor
    - Linear Regression
    - Ridge Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Model evaluation using **MAE, RMSE, and R²**
    - **5-Fold Cross Validation**
    - **Hyperparameter tuning using GridSearchCV**

    ### CRISP-DM Framework
    This project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology:

    **Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment**

    ### Deployment Prototype
    The final prototype was implemented using **Streamlit**, allowing users to:
    - explore the dataset
    - analyse model performance
    - generate productivity predictions interactively

    ### Business Insight
    The predictive model can help identify whether a production team is likely to **meet or fall short of its productivity target**, enabling more **data-driven decision making in manufacturing operations**.

    """)
