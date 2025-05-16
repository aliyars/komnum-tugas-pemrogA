import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import format_polynomial

def train_population_model(pop_years, pop_values):
    """
    Fungsi untuk melatih model regresi polinomial untuk data populasi.
    
    Fungsi ini akan mencoba beberapa derajat polinomial (2-5) dan memilih
    yang terbaik berdasarkan skor R².
    
    Parameter:
        pop_years: Array tahun untuk data populasi
        pop_values: Array nilai populasi
    Return:
        Dictionary berisi model terbaik dan informasi terkait
    """
    best_pop_degree = 0
    best_pop_r2 = 0
    best_model = None
    best_poly_features = None
    
    # Mencoba berbagai derajat polinomial untuk menemukan yang terbaik
    for degree in range(2, 6):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(pop_years)
        
        model = LinearRegression()
        model.fit(X_poly, pop_values)
        
        y_pred = model.predict(X_poly)
        r2 = r2_score(pop_values, y_pred)
        
        print(f"Derajat {degree}: R² = {r2}")
        
        if r2 > best_pop_r2:
            best_pop_degree = degree
            best_pop_r2 = r2
            best_model = model
            best_poly_features = poly_features
    
    print(f"Derajat polinomial terbaik untuk populasi: {best_pop_degree} dengan R² = {best_pop_r2}")
    
    # Mendapatkan koefisien untuk model terbaik
    pop_coeffs = best_model.coef_
    pop_intercept = best_model.intercept_
    
    # Membuat persamaan
    pop_equation = format_polynomial(pop_coeffs, pop_intercept, best_pop_degree)
    print("Persamaan polinomial populasi:")
    print(pop_equation)
    
    # Mengembalikan informasi model
    return {
        'model': best_model,
        'poly_features': best_poly_features,
        'degree': best_pop_degree,
        'r2': best_pop_r2,
        'coeffs': pop_coeffs,
        'intercept': pop_intercept,
        'equation': pop_equation
    }