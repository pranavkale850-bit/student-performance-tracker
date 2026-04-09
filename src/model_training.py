import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully")
        return df
    except FileNotFoundError:
        print("❌ File not found. Check path.")
        exit()
    except Exception as e:
        print("❌ Error loading data:", e)
        exit()


# -------------------------------
# EDA
# -------------------------------
def perform_eda(df):
    try:
        sns.countplot(x="performance", data=df)
        plt.title("Distribution of Performance")
        plt.show()
    except Exception as e:
        print("⚠️ EDA error:", e)


# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_model(df):
    try:
        X = df.drop("performance", axis=1)
        y = df["performance"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Reset index for alignment
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        print("✅ Model trained successfully")

        return model, X_test, y_test, y_pred, X

    except Exception as e:
        print("❌ Training error:", e)
        exit()


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(y_test, y_pred):
    try:
        print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
        print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Average", "Good", "Poor"],
                    yticklabels=["Average", "Good", "Poor"])

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    except Exception as e:
        print("⚠️ Evaluation error:", e)


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
def plot_feature_importance(model, X):
    try:
        importances = model.feature_importances_

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\n🔝 Top Features:\n", feature_importance.head(5))

        feature_importance.head(10).plot(
            x="Feature", y="Importance", kind="barh", figsize=(8, 5)
        )

        plt.title("Top Features Affecting Performance")
        plt.show()

    except Exception as e:
        print("⚠️ Feature importance error:", e)


# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
def give_recommendation(row):
    suggestions = []

    try:
        if row.get("studytime", 0) <= 2:
            suggestions.append("Increase study time")

        if row.get("absences", 0) > 5:
            suggestions.append("Reduce absences")

        if row.get("goout", 0) > 3:
            suggestions.append("Reduce going out")

        if row.get("Walc", 0) > 2:
            suggestions.append("Reduce alcohol consumption")

        if row.get("health", 3) < 3:
            suggestions.append("Improve health")

        return suggestions if suggestions else ["Maintain current performance"]

    except Exception as e:
        return ["Error generating recommendation"]


# -------------------------------
# APPLY RECOMMENDATIONS
# -------------------------------
def generate_results(X_test, y_test, y_pred):
    try:
        results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        results["Recommendation"] = results.apply(
            lambda row: give_recommendation(X_test.iloc[row.name])
            if row["Predicted"] == "Poor" else "No major issues",
            axis=1
        )

        print("\n📌 Sample Results:\n", results.head(10))
        return results

    except Exception as e:
        print("⚠️ Recommendation error:", e)


# -------------------------------
# PREDICT NEW STUDENT
# -------------------------------
def predict_student(model, input_data):
    try:
        df_input = pd.DataFrame([input_data])
        pred = model.predict(df_input)[0]
        rec = give_recommendation(df_input.iloc[0])

        print("\n🎯 Prediction:", pred)
        print("💡 Recommendations:", rec)

    except Exception as e:
        print("⚠️ Prediction error:", e)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    df = load_data("clean_data.csv")

    perform_eda(df)

    model, X_test, y_test, y_pred, X = train_model(df)

    evaluate_model(y_test, y_pred)

    plot_feature_importance(model, X)

    results = generate_results(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()


