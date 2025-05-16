import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from utils import format_logistic

def logistic_function(x, L, k, x0):
    """
    Fungsi logistik dengan parameter:
    L = nilai asimtot atas (batas maksimal, dalam kasus ini 100%)
    k = tingkat pertumbuhan
    x0 = titik tengah kurva (saat nilai fungsi = L/2)
    """
    return L / (1 + np.exp(-k * (x - x0)))

def train_internet_model(internet_years, internet_values):
    """
    Fungsi untuk melatih model logistik untuk data penggunaan internet.
    
    Fungsi ini menggunakan curve_fit dari scipy untuk menemukan parameter
    optimal dari fungsi logistik yang cocok dengan data persentase pengguna internet.
    
    Parameter:
        internet_years: Array tahun untuk data internet
        internet_values: Array nilai persentase pengguna internet
    Return:
        Dictionary berisi parameter model dan informasi terkait
    """
    # Parameter awal untuk model logistik
    # L=100 (batas atas 100%)
    # k=0.3 (tingkat pertumbuhan yang diestimasi)
    # x0=2010 (tahun di mana pertumbuhan paling cepat, sekitar 2010-an)
    p0 = [100, 0.3, 2010]
    
    # Batasan parameter untuk menghasilkan model yang masuk akal
    # L harus antara 0 dan 100 (persentase)
    # k harus positif
    # x0 harus dalam rentang tahun data
    bounds = ([0, 0, 1990], [100, 1, 2020])
    
    # Melakukan fitting model logistik
    params, covariance = curve_fit(logistic_function, internet_years, internet_values, 
                                  p0=p0, bounds=bounds, maxfev=10000)
    
    # Mengambil parameter hasil fitting
    L_fit, k_fit, x0_fit = params
    print(f"Parameter model logistik: L={L_fit:.2f}, k={k_fit:.4f}, x0={x0_fit:.2f}")
    
    # Membuat persamaan
    internet_equation = format_logistic(L_fit, k_fit, x0_fit)
    print("Persamaan logistik persentase pengguna internet:")
    print(internet_equation)
    
    # Menghitung R^2 untuk model logistik
    internet_pred = logistic_function(internet_years, L_fit, k_fit, x0_fit)
    internet_r2 = r2_score(internet_values, internet_pred)
    print(f"R² model logistik untuk internet: {internet_r2:.4f}")
    
    # Mengembalikan informasi model
    return {
        'params': params,
        'L': L_fit,
        'k': k_fit,
        'x0': x0_fit,
        'equation': internet_equation,
        'r2': internet_r2,
        'function': logistic_function
    }