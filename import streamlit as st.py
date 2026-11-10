import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import date

 

# ---------------------------------------------------------

# HOSPITAL PATIENT RECOVERY ANALYSIS SYSTEM

# ---------------------------------------------------------

 

st.set_page_config(

    page_title="Hospital Patient Recovery Analysis",

    page_icon="🏥",

    layout="wide"

)

 

# Simple healthcare-style appearance

st.markdown("""

<style>

    .stApp { background-color: #f7fbfd; }

    h1, h2, h3 { color: #155b75; }

    div[data-testid="stMetric"] {

        background-color: #eaf5f8;

        border: 1px solid #d2e9ef;

        padding: 14px;

        border-radius: 10px;

    }

</style>

""", unsafe_allow_html=True)

 

st.title("🏥 Hospital Patient Recovery Analysis System")

st.write(

    "This application analyses hospital patient records, cleans the data, "

    "calculates recovery statistics, and displays charts and observations."

)

 

st.info(

    "Learning project only: the sample records are fictional. "

    "This application is not a medical decision-making tool."

)

 

# Sample dataset is available immediately, even without uploading a file.

sample_data = pd.DataFrame([

    [201, "Aarav", "2024-06-10", "1988-03-15", 95, 90],

    [202, "Meera", "2024-06-12", "1995-08-20", 82, 78],

    [203, "Kavin", "2024-06-15", "1979-01-10", 90, 88],

    [204, "Diya", "2024-07-01", "2001-11-25", 75, 70],

    [205, "Rohan", "2024-07-05", "1968-05-18", 88, 84],

    [206, "Nila", "2024-07-10", "1990-09-12", 60, 58],

    [207, "Ishaan", "2024-08-01", "1985-02-22", 92, 94],

    [208, "Ananya", "2024-08-05", "2003-12-30", 85, 80],

    [209, "Vikram", "2024-08-10", "1972-04-05", 70, 65],

    [210, "Sara", "2024-08-15", "1998-06-17", 98, 96],

], columns=[

    "Patient_ID", "Name", "Admission_Date", "DOB",

    "Treatment_Adherence (%)", "Recovery_Score"

])

 

required_columns = [

    "Patient_ID", "Name", "Admission_Date", "DOB",

    "Treatment_Adherence (%)", "Recovery_Score"

]

 

st.sidebar.header("Data Source")

data_choice = st.sidebar.radio(

    "Choose patient data",

    ["Use sample dataset", "Upload my own CSV"]

)

 

uploaded_file = None

if data_choice == "Upload my own CSV":

    uploaded_file = st.sidebar.file_uploader(

        "Upload a CSV file",

        type=["csv"]

    )

 

reference_date = st.sidebar.date_input(

    "Reference date for date calculations",

    value=date.today()

)

 

if data_choice == "Upload my own CSV" and uploaded_file is None:

    st.warning("Please upload a CSV file, or select 'Use sample dataset'.")

    st.stop()

 

# Read selected data

try:

    if data_choice == "Use sample dataset":

        original_df = sample_data.copy()

    else:

        original_df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Could not read the CSV file: {error}")

    st.stop()

 

st.header("1. Original Patient Records")

st.dataframe(original_df, use_container_width=True)

 

# Check required columns

missing_columns = [col for col in required_columns if col not in original_df.columns]

if missing_columns:

    st.error(

        "The uploaded CSV is missing these required columns: "

        + ", ".join(missing_columns)

    )

    st.write("Required column names:", required_columns)

    st.stop()

 

# ---------------------------------------------------------

# 2. DATA PRE-PROCESSING

# ---------------------------------------------------------

st.header("2. Data Pre-processing")

 

df = original_df.copy()

original_rows = len(df)

 

# Remove completely repeated rows

duplicate_rows = int(df.duplicated().sum())

df = df.drop_duplicates().copy()

 

# Clean text fields

df["Name"] = df["Name"].astype("string").str.strip()

df["Name"] = df["Name"].replace("", pd.NA).fillna("Unknown")

 

# Convert dates and numerical columns

df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")

df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")

 

numeric_columns = [

    "Patient_ID", "Treatment_Adherence (%)", "Recovery_Score"

]

for column in numeric_columns:

    df[column] = pd.to_numeric(df[column], errors="coerce")

 

# Values outside the allowed 0-100 range are treated as missing.

for column in ["Treatment_Adherence (%)", "Recovery_Score"]:

    invalid_count = int((~df[column].between(0, 100) & df[column].notna()).sum())

    df.loc[~df[column].between(0, 100), column] = np.nan

 

missing_before_fill = int(df.isna().sum().sum())

 

# Fill missing numeric values with the median, if a valid median exists.

for column in ["Treatment_Adherence (%)", "Recovery_Score"]:

    median_value = df[column].median()

    if pd.notna(median_value):

        df[column] = df[column].fillna(median_value)

 

# Essential fields must be available for a record to be analysed.

before_essential_drop = len(df)

df = df.dropna(

    subset=["Patient_ID", "Admission_Date", "DOB"]

).copy()

essential_rows_removed = before_essential_drop - len(df)

 

# Keep only the first record for each patient ID.

before_id_dedup = len(df)

df = df.drop_duplicates(subset=["Patient_ID"], keep="first").copy()

duplicate_ids_removed = before_id_dedup - len(df)

 

# Remove records if numeric values could not be filled (e.g. entire column empty).

df = df.dropna(

    subset=["Treatment_Adherence (%)", "Recovery_Score"]

).copy()

 

if df.empty:

    st.error("No valid patient records remain after cleaning. Check the CSV data.")

    st.stop()

 

df["Patient_ID"] = df["Patient_ID"].astype(int)

 

col_a, col_b, col_c = st.columns(3)

col_a.metric("Original records", original_rows)

col_b.metric("Exact duplicate rows removed", duplicate_rows)

col_c.metric("Records after cleaning", len(df))

 

st.write("Missing values in the original uploaded data:", int(original_df.isna().sum().sum()))

st.write("Missing values detected during cleaning:", missing_before_fill)

st.write("Records removed due to missing essential information:", essential_rows_removed)

st.write("Repeated patient IDs removed:", duplicate_ids_removed)

 

st.subheader("Cleaned Patient Records")

st.dataframe(df, use_container_width=True)

 

# ---------------------------------------------------------

# 3. DATE AND TIME MANIPULATION

# ---------------------------------------------------------

st.header("3. Date and Time Analysis")

 

ref_timestamp = pd.Timestamp(reference_date)

 

df["Admission_Month"] = df["Admission_Date"].dt.month_name()

df["Admission_Year"] = df["Admission_Date"].dt.year

 

birthday_not_reached = (

    (df["DOB"].dt.month > ref_timestamp.month)

    | (

        (df["DOB"].dt.month == ref_timestamp.month)

        & (df["DOB"].dt.day > ref_timestamp.day)

    )

)

 

df["Age"] = (

    ref_timestamp.year

    - df["DOB"].dt.year

    - birthday_not_reached.astype(int)

)

 

df["Days_Since_Admission"] = (

    ref_timestamp - df["Admission_Date"]

).dt.days

 

date_columns = [

    "Name", "Admission_Date", "Admission_Month",

    "Admission_Year", "DOB", "Age", "Days_Since_Admission"

]

st.dataframe(df[date_columns], use_container_width=True)

 

# ---------------------------------------------------------

# 4. EXPLORATORY DATA ANALYSIS

# ---------------------------------------------------------

st.header("4. Exploratory Data Analysis (EDA)")

 

recovery = df["Recovery_Score"].to_numpy(dtype=float)

adherence = df["Treatment_Adherence (%)"].to_numpy(dtype=float)

 

average_recovery = float(np.mean(recovery))

median_recovery = float(np.median(recovery))

highest_recovery = float(np.max(recovery))

lowest_recovery = float(np.min(recovery))

average_adherence = float(np.mean(adherence))

 

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total patients", len(df))

m2.metric("Average recovery", f"{average_recovery:.2f}")

m3.metric("Median recovery", f"{median_recovery:.2f}")

m4.metric("Average adherence", f"{average_adherence:.2f}%")

 

m5, m6 = st.columns(2)

m5.metric("Highest recovery score", f"{highest_recovery:.0f}")

m6.metric("Lowest recovery score", f"{lowest_recovery:.0f}")

 

st.subheader("Descriptive Statistics")

st.dataframe(

    df[["Treatment_Adherence (%)", "Recovery_Score", "Age"]].describe(),

    use_container_width=True

)

 

st.subheader("Highest Recovery Patient(s)")

st.dataframe(

    df.loc[df["Recovery_Score"] == highest_recovery,

           ["Patient_ID", "Name", "Recovery_Score"]],

    use_container_width=True

)

 

st.subheader("Lowest Recovery Patient(s)")

st.dataframe(

    df.loc[df["Recovery_Score"] == lowest_recovery,

           ["Patient_ID", "Name", "Recovery_Score"]],

    use_container_width=True

)

 

st.subheader("Patients Admitted by Month")

month_order = [

    "January", "February", "March", "April", "May", "June",

    "July", "August", "September", "October", "November", "December"

]

monthly_counts = df.groupby("Admission_Month").size()

monthly_counts = monthly_counts.reindex(

    [month for month in month_order if month in monthly_counts.index]

)

st.dataframe(

    monthly_counts.rename("Patient Count").reset_index(),

    use_container_width=True

)

 

# ---------------------------------------------------------

# 5. DATA VISUALIZATION

# ---------------------------------------------------------

st.header("5. Data Visualization")

 

# Bar chart

fig1, ax1 = plt.subplots(figsize=(9, 4))

ax1.bar(df["Name"], df["Recovery_Score"])

ax1.set_title("Patient Recovery Scores")

ax1.set_xlabel("Patient")

ax1.set_ylabel("Recovery Score")

ax1.set_ylim(0, 100)

ax1.tick_params(axis="x", rotation=35)

fig1.tight_layout()

st.pyplot(fig1)

plt.close(fig1)

 

# Histogram

fig2, ax2 = plt.subplots(figsize=(9, 4))

ax2.hist(df["Recovery_Score"], bins=5, edgecolor="black")

ax2.set_title("Distribution of Recovery Scores")

ax2.set_xlabel("Recovery Score")

ax2.set_ylabel("Number of Patients")

fig2.tight_layout()

st.pyplot(fig2)

plt.close(fig2)

 

# Scatter plot

fig3, ax3 = plt.subplots(figsize=(9, 4))

ax3.scatter(df["Treatment_Adherence (%)"], df["Recovery_Score"])

ax3.set_title("Treatment Adherence vs Recovery Score")

ax3.set_xlabel("Treatment Adherence (%)")

ax3.set_ylabel("Recovery Score")

ax3.set_xlim(0, 100)

ax3.set_ylim(0, 100)

fig3.tight_layout()

st.pyplot(fig3)

plt.close(fig3)

 

# ---------------------------------------------------------

# 6. INTERPRETATION

# ---------------------------------------------------------

st.header("6. Interpretation of Results")

 

st.write(

    f"The average recovery score is {average_recovery:.2f} out of 100. "

    f"The average treatment adherence is {average_adherence:.2f}%."

)

 

if len(df) >= 2 and np.std(adherence) > 0 and np.std(recovery) > 0:

    correlation = float(np.corrcoef(adherence, recovery)[0, 1])

    st.write(f"Correlation coefficient: {correlation:.3f}")

 

    if correlation > 0.5:

        st.success(

            "The sample shows a positive association between treatment "

            "adherence and recovery scores."

        )

    elif correlation < -0.5:

        st.warning(

            "The sample shows a negative association between treatment "

            "adherence and recovery scores."

        )

    else:

        st.info(

            "The sample shows a weak or limited linear association between "

            "treatment adherence and recovery scores."

        )

else:

    st.info("Correlation cannot be calculated for these records.")

 

best_patient = df.loc[df["Recovery_Score"].idxmax(), "Name"]

lowest_patient = df.loc[df["Recovery_Score"].idxmin(), "Name"]

 

st.write(f"Highest recovery score in this dataset: {best_patient}.")

st.write(f"Lowest recovery score in this dataset: {lowest_patient}.")

st.caption(

    "These are descriptive patterns in the supplied data. Correlation does "

    "not prove that treatment adherence causes recovery. This tool is not "

    "intended for clinical decisions."

)

 

# ---------------------------------------------------------

# 7. DOWNLOAD CLEANED DATA

# ---------------------------------------------------------

st.header("7. Download Cleaned Dataset")

 

csv_data = df.to_csv(index=False).encode("utf-8")

 

st.download_button(

    label="Download Cleaned Patient Dataset",

    data=csv_data,

    file_name="cleaned_hospital_patients.csv",

    mime="text/csv"

)

 

st.success("Analysis completed successfully.")