import pandas as pd
import sweetviz as sv

def run_eda(file_path):
    print("📂 Loading dataset...")
    df = pd.read_csv(file_path)

    print("✅ Dataset Loaded!")
    print("Shape:", df.shape)

    print("📊 Generating Sweetviz Report...")
    report = sv.analyze(df)

    report.show_html("eda_report.html")

    print("✅ Report Generated Successfully!")
    print("Open: eda_report.html")


# Example Run
if __name__ == "__main__":
    run_eda(r"C:\Users\marve\OneDrive\Desktop\dataanalysi\project8\data_agent\data.csv")  
    # Replace with your dataset file
