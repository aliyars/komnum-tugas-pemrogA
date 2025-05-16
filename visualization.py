import matplotlib.pyplot as plt
import numpy as np

def plot_population_data(pop_years, pop_values):
    """
    Fungsi untuk memvisualisasikan data populasi.
    
    Parameter:
        pop_years: Array tahun untuk data populasi
        pop_values: Array nilai populasi
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(pop_years, pop_values, color='blue', label='Data Aktual')
    plt.title('Pertumbuhan Populasi Indonesia (1960-2023)')
    plt.xlabel('Tahun')
    plt.ylabel('Populasi')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_internet_data(internet_years, internet_values):
    """
    Fungsi untuk memvisualisasikan data penggunaan internet.
    
    Parameter:
        internet_years: Array tahun untuk data internet
        internet_values: Array nilai persentase pengguna internet
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(internet_years, internet_values, color='red', label='Data Aktual')
    plt.title('Persentase Pengguna Internet di Indonesia (1960-2023)')
    plt.xlabel('Tahun')
    plt.ylabel('Persentase Pengguna Internet (%)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_internet_model(internet_years, internet_values, logistic_function, params, years_range=None):
    """
    Fungsi untuk memvisualisasikan data penggunaan internet dengan model logistik.
    
    Parameter:
        internet_years: Array tahun untuk data internet
        internet_values: Array nilai persentase pengguna internet
        logistic_function: Fungsi logistik untuk memprediksi nilai
        params: Parameter model logistik [L, k, x0]
        years_range: Rentang tahun untuk visualisasi (opsional)
    """
    L_fit, k_fit, x0_fit = params
    
    plt.figure(figsize=(12, 6))
    plt.scatter(internet_years, internet_values, color='red', label='Data Aktual')
    
    # Membuat kurva halus untuk plot
    if years_range is None:
        years_range = np.linspace(min(internet_years), 2035, 100)
    
    internet_curve = logistic_function(years_range, L_fit, k_fit, x0_fit)
    
    plt.plot(years_range, internet_curve, 'b-', label='Model Logistik')
    plt.title('Model Pertumbuhan Persentase Pengguna Internet di Indonesia')
    plt.xlabel('Tahun')
    plt.ylabel('Persentase Pengguna Internet (%)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_population_with_estimates(pop_years, pop_values, pop_model_info, missing_years, estimated_population, 
                                future_years, predicted_population, all_years):
    """
    Fungsi untuk memvisualisasikan data populasi dengan estimasi dan prediksi.
    
    Parameter:
        pop_years: Array tahun untuk data populasi aktual
        pop_values: Array nilai populasi aktual
        pop_model_info: Dictionary berisi informasi model populasi
        missing_years: Array tahun dengan nilai yang hilang
        estimated_population: Array nilai populasi yang diestimasi
        future_years: Array tahun untuk prediksi masa depan
        predicted_population: Array nilai populasi yang diprediksi
        all_years: Array semua tahun dalam dataset
    """
    plt.figure(figsize=(14, 7))
    
    # Plot data asli
    plt.scatter(pop_years, pop_values, color='blue', label='Data Populasi Aktual')
    
    # Membuat data untuk kurva
    years_range = np.linspace(min(all_years), 2035, 100).reshape(-1, 1)
    years_range_poly_pop = pop_model_info['poly_features'].transform(years_range)
    pop_curve = pop_model_info['model'].predict(years_range_poly_pop)
    
    # Plot kurva regresi
    plt.plot(years_range, pop_curve, 'g-', label=f'Model Polinomial (Derajat {pop_model_info["degree"]})')
    
    # Plot nilai yang diestimasi
    plt.scatter(missing_years, estimated_population, color='red', s=100, marker='x', label='Nilai Populasi yang Diestimasi')
    
    # Plot nilai prediksi
    plt.scatter(future_years, predicted_population, color='purple', s=100, marker='*', label='Nilai Populasi yang Diprediksi')
    
    plt.title('Model Pertumbuhan Populasi Indonesia')
    plt.xlabel('Tahun')
    plt.ylabel('Populasi')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_internet_with_estimates(internet_years, internet_values, logistic_function, params, 
                                missing_years_array, estimated_internet, future_years_array, 
                                predicted_internet_percentage, all_years):
    """
    Fungsi untuk memvisualisasikan data penggunaan internet dengan estimasi dan prediksi.
    
    Parameter:
        internet_years: Array tahun untuk data internet aktual
        internet_values: Array nilai persentase pengguna internet aktual
        logistic_function: Fungsi logistik untuk memprediksi nilai
        params: Parameter model logistik [L, k, x0]
        missing_years_array: Array tahun dengan nilai yang hilang
        estimated_internet: Array nilai persentase internet yang diestimasi
        future_years_array: Array tahun untuk prediksi masa depan
        predicted_internet_percentage: Array nilai persentase internet yang diprediksi
        all_years: Array semua tahun dalam dataset
    """
    L_fit, k_fit, x0_fit = params
    
    plt.figure(figsize=(14, 7))
    
    # Plot data asli
    plt.scatter(internet_years, internet_values, color='red', label='Data Persentase Internet Aktual')
    
    # Membuat data untuk kurva
    years_range_flat = np.linspace(min(all_years), 2035, 100)
    internet_curve = logistic_function(years_range_flat, L_fit, k_fit, x0_fit)
    
    # Plot kurva logistik
    plt.plot(years_range_flat, internet_curve, 'b-', label='Model Logistik')
    
    # Plot nilai yang diestimasi
    plt.scatter(missing_years_array, estimated_internet, color='green', s=100, marker='x', label='Nilai Persentase Internet yang Diestimasi')
    
    # Plot nilai prediksi
    plt.scatter(future_years_array, predicted_internet_percentage, color='orange', s=100, marker='*', label='Nilai Persentase Internet yang Diprediksi')
    
    plt.title('Model Pertumbuhan Persentase Pengguna Internet di Indonesia')
    plt.xlabel('Tahun')
    plt.ylabel('Persentase Pengguna Internet (%)')
    plt.ylim(0, 105)  # Memastikan visualisasi menunjukkan batas maksimum 100%
    plt.grid(True)
    plt.legend()
    plt.show()