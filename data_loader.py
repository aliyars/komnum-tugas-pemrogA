import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Fungsi untuk memuat dan memproses data.
    Parameter:
        file_path: Path ke file CSV
    Return:
        Dictionary berisi data yang sudah diproses
    """
    # Memuat data dari file CSV
    data = pd.read_csv(file_path)
    
    # Menampilkan informasi data
    print("Data head:")
    print(data.head())
    print("\nInformasi data:")
    print(data.info())
    print("\nDeskripsi statistik data:")
    print(data.describe())
    
    # Memeriksa nilai yang hilang
    print("\nNilai yang hilang:")
    print(data.isnull().sum())
    print("Tahun dengan nilai yang hilang:")
    print(data[data.isnull().any(axis=1)]['Year'])
    
    # Mempersiapkan dataset untuk analisis
    # Dataset populasi: menghilangkan baris dengan nilai populasi yang hilang
    pop_data = data.dropna(subset=['Population'])
    pop_years = pop_data['Year'].values.reshape(-1, 1)
    pop_values = pop_data['Population'].values
    
    # Dataset internet: menghilangkan baris dengan nilai persentase internet yang hilang
    internet_data = data.dropna(subset=['Percentage_Internet_User'])
    internet_years = internet_data['Year'].values
    internet_values = internet_data['Percentage_Internet_User'].values
    
    # Tahun-tahun dengan nilai yang hilang
    missing_years = np.array([2005, 2006, 2015, 2016]).reshape(-1, 1)
    missing_years_array = np.array([2005, 2006, 2015, 2016])
    all_years = data['Year'].values
    all_years_reshape = all_years.reshape(-1, 1)
    
    # Mengembalikan dictionary berisi semua data yang telah diproses
    return {
        'data': data,
        'pop_data': pop_data,
        'pop_years': pop_years,
        'pop_values': pop_values,
        'internet_data': internet_data,
        'internet_years': internet_years,
        'internet_values': internet_values,
        'missing_years': missing_years,
        'missing_years_array': missing_years_array,
        'all_years': all_years,
        'all_years_reshape': all_years_reshape
    }