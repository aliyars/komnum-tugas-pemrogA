import numpy as np
import pandas as pd

from data_loader import load_data
from population_model import train_population_model
from internet_usage_model import train_internet_model
from visualization import (plot_population_data, plot_internet_data, plot_internet_model,
                          plot_population_with_estimates, plot_internet_with_estimates)
from estimation import estimate_missing_values, predict_future_values, print_summary

def main():
    """
    Fungsi utama yang menjalankan seluruh alur analisis data.
    
    Alur:
    1. Memuat dan memproses data
    2. Memvisualisasikan data asli
    3. Melatih model untuk populasi dan penggunaan internet
    4. Memvisualisasikan model
    5. Mengestimasi nilai yang hilang
    6. Memprediksi nilai masa depan
    7. Memvisualisasikan hasil dengan estimasi dan prediksi
    8. Mencetak ringkasan hasil
    """
    # Memuat dan memproses data
    data_dict = load_data('Data Tugas Pemrograman A.csv')
    
    # Memvisualisasikan data asli
    plot_population_data(data_dict['pop_years'], data_dict['pop_values'])
    plot_internet_data(data_dict['internet_years'], data_dict['internet_values'])
    
    # Melatih model
    pop_model_info = train_population_model(data_dict['pop_years'], data_dict['pop_values'])
    internet_model_info = train_internet_model(data_dict['internet_years'], data_dict['internet_values'])
    
    # Memvisualisasikan model internet
    plot_internet_model(data_dict['internet_years'], data_dict['internet_values'], 
                       internet_model_info['function'], 
                       [internet_model_info['L'], internet_model_info['k'], internet_model_info['x0']])
    
    # Mengestimasi nilai yang hilang
    estimation_results = estimate_missing_values(pop_model_info, internet_model_info, 
                                               data_dict['missing_years'], 
                                               data_dict['missing_years_array'])
    
    # Memprediksi nilai masa depan
    future_years_array = np.array([2030, 2035])
    prediction_results = predict_future_values(pop_model_info, internet_model_info, future_years_array)
    
    # Memvisualisasikan dengan estimasi dan prediksi
    plot_population_with_estimates(data_dict['pop_years'], data_dict['pop_values'], 
                                  pop_model_info, data_dict['missing_years'], 
                                  estimation_results['estimated_population'], 
                                  future_years_array.reshape(-1, 1), 
                                  prediction_results['predicted_population'], 
                                  data_dict['all_years'])
    
    plot_internet_with_estimates(data_dict['internet_years'], data_dict['internet_values'], 
                                internet_model_info['function'], 
                                [internet_model_info['L'], internet_model_info['k'], internet_model_info['x0']], 
                                data_dict['missing_years_array'], 
                                estimation_results['estimated_internet'], 
                                future_years_array, 
                                prediction_results['predicted_internet_percentage'], 
                                data_dict['all_years'])
    
    # Mencetak ringkasan
    print_summary(pop_model_info, internet_model_info, estimation_results, prediction_results)

if __name__ == "__main__":
    main()