def schwefel_1_2(x):
    D = len(x)
    total_sum = 0
    for i in range(D):
        inner_sum = sum(x[:i+1])  # Somatório de x_j para j=1 até i
        total_sum += inner_sum ** 2
    return total_sum

def shifted_rosenbrock(z):
    D = len(z)
    total = 0
    for i in range(D - 1):
        total += 100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
    return total