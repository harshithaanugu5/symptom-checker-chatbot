import pandas as pd

def load_and_clean(path="data/clinical_notes_dataset.csv"):
    df = pd.read_csv(path)

    # Combine symptoms into one field
    df["Symptoms"] = (
        df["Primary_Complaint"].fillna("") + " " +
        df["Additional_Symptoms"].fillna("")
    ).str.lower()

    # Remove missing suggestions
    df["Diagnosis_Suggestion"] = df["Diagnosis_Suggestion"].fillna("Unknown")

    return df
