import pandas as pd
import numpy as np

def estimate_missing_values(pop_model_info, internet_model_info, missing_years, missing_years_array):
    """
    Fungsi untuk mengestimasi nilai yang hilang untuk populasi dan penggunaan internet.
    
    Menggunakan model polinomial untuk mengestimasi populasi dan model logistik untuk
    mengestimasi persentase pengguna internet.
    
    Parameter:
        pop_model_info: Dictionary berisi informasi model populasi
        internet_model_info: Dictionary berisi informasi model internet
        missing_years: Array tahun dengan nilai yang hilang (reshape untuk model populasi)
        missing_years_array: Array tahun dengan nilai yang hilang (flat untuk model internet)
    Return:
        Dictionary berisi hasil estimasi
    """
    # Estimasi populasi untuk tahun yang hilang (menggunakan model polinomial)
    missing_years_poly_pop = pop_model_info['poly_features'].transform(missing_years)
    estimated_population = pop_model_info['model'].predict(missing_years_poly_pop)
    
    # Estimasi persentase internet untuk tahun yang hilang (menggunakan model logistik)
    estimated_internet = internet_model_info['function'](missing_years_array, 
                                                      internet_model_info['L'], 
                                                      internet_model_info['k'], 
                                                      internet_model_info['x0'])
    
    # Hasil estimasi
    results = pd.DataFrame({
        'Year': missing_years_array,
        'Estimated_Population': estimated_population,
        'Estimated_Internet_Percentage': estimated_internet
    })
    print("\nHasil Estimasi Nilai yang Hilang:")
    print(results)
    
    return {
        'estimated_population': estimated_population,
        'estimated_internet': estimated_internet,
        'results_df': results
    }

def predict_future_values(pop_model_info, internet_model_info, future_years_array):
    """
    Fungsi untuk memprediksi nilai masa depan untuk populasi dan penggunaan internet.
    
    Menggunakan model polinomial untuk memprediksi populasi dan model logistik untuk
    memprediksi persentase pengguna internet.
    
    Parameter:
        pop_model_info: Dictionary berisi informasi model populasi
        internet_model_info: Dictionary berisi informasi model internet
        future_years_array: Array tahun untuk prediksi masa depan
    Return:
        Dictionary berisi hasil prediksi
    """
    future_years = future_years_array.reshape(-1, 1)
    
    # Prediksi populasi (menggunakan model polinomial)
    future_years_poly_pop = pop_model_info['poly_features'].transform(future_years)
    predicted_population = pop_model_info['model'].predict(future_years_poly_pop)
    
    # Prediksi persentase internet (menggunakan model logistik)
    predicted_internet_percentage = internet_model_info['function'](future_years_array, 
                                                                 internet_model_info['L'], 
                                                                 internet_model_info['k'], 
                                                                 internet_model_info['x0'])
    
    # Menghitung jumlah pengguna internet
    predicted_internet_users = (predicted_internet_percentage / 100) * predicted_population
    
    # Hasil prediksi
    future_results = pd.DataFrame({
        'Year': future_years_array,
        'Predicted_Population': predicted_population,
        'Predicted_Internet_Percentage': predicted_internet_percentage,
        'Predicted_Internet_Users': predicted_internet_users
    })
    print("\nHasil Prediksi untuk Tahun 2030 dan 2035:")
    print(future_results)
    
    return {
        'predicted_population': predicted_population,
        'predicted_internet_percentage': predicted_internet_percentage,
        'predicted_internet_users': predicted_internet_users,
        'results_df': future_results
    }

def print_summary(pop_model_info, internet_model_info, estimation_results, prediction_results):
    """
    Fungsi untuk mencetak ringkasan hasil analisis.
    
    Parameter:
        pop_model_info: Dictionary berisi informasi model populasi
        internet_model_info: Dictionary berisi informasi model internet
        estimation_results: Dictionary berisi hasil estimasi nilai yang hilang
        prediction_results: Dictionary berisi hasil prediksi nilai masa depan
    """
    print("\nRINGKASAN HASIL ANALISIS:")
    print("==========================")
    print(f"1. Model Populasi: Polinomial derajat {pop_model_info['degree']} (R² = {pop_model_info['r2']:.4f})")
    print(f"   Persamaan: {pop_model_info['equation']}")
    print(f"2. Model Internet: Logistik (R² = {internet_model_info['r2']:.4f})")
    print(f"   Persamaan: {internet_model_info['equation']}")
    
    missing_years_array = estimation_results['results_df']['Year'].values
    estimated_population = estimation_results['estimated_population']
    estimated_internet = estimation_results['estimated_internet']
    
    print("\nEstimasi Nilai Hilang:")
    for i, year in enumerate(missing_years_array):
        print(f"   {year}: Populasi = {int(estimated_population[i]):,} jiwa, Internet = {estimated_internet[i]:.2f}%")
    
    future_years_array = prediction_results['results_df']['Year'].values
    predicted_population = prediction_results['predicted_population']
    predicted_internet_percentage = prediction_results['predicted_internet_percentage']
    predicted_internet_users = prediction_results['predicted_internet_users']
    
    print("\nPrediksi Masa Depan:")
    for i, year in enumerate(future_years_array):
        print(f"   {year}: Populasi = {int(predicted_population[i]):,} jiwa, Internet = {predicted_internet_percentage[i]:.2f}%, Pengguna = {int(predicted_internet_users[i]):,} jiwa")