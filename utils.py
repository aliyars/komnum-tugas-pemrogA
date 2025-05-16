def format_polynomial(coeffs, intercept, degree):
    """
    Fungsi untuk memformat persamaan polinomial menjadi string yang mudah dibaca.
    Parameter:
        coeffs: Koefisien polinomial
        intercept: Intercept model
        degree: Derajat polinomial
    Return:
        String persamaan polinomial
    """
    equation = f"y = {intercept:.2f}"
    for i in range(1, degree + 1):
        if coeffs[i] >= 0:
            equation += f" + {coeffs[i]:.6f}x^{i}"
        else:
            equation += f" - {abs(coeffs[i]):.6f}x^{i}"
    return equation

def format_logistic(L, k, x0):
    """
    Fungsi untuk memformat persamaan logistik menjadi string yang mudah dibaca.
    Parameter:
        L: Nilai asimtot atas (batas maksimal)
        k: Tingkat pertumbuhan
        x0: Titik tengah kurva
    Return:
        String persamaan logistik
    """
    return f"y = {L:.2f} / (1 + exp(-{k:.6f} * (x - {x0:.2f})))"