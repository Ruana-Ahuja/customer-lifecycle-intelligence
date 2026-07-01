import pandas as pd


def load_data(filepath):
    return pd.read_excel(filepath, sheet_name='Year 2010-2011')


def clean_data(df):
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df = df.drop_duplicates()

    df['TotalPrice'] = df['Quantity'] * df['Price']
    df = df.reset_index(drop=True)

    return df


def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

