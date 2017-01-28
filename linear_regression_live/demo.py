import numpy as np
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept
def compute_error(b, m, points):
    error_total = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        
        #soustraire le 'vrai' y  de celui estimé par m et b ( (m * x + b) )
        error_total += (y  - (m * x + b)) ** 2

        # retourner la moyenne d'erreur
    return error_total/ float(len(points))


def gradient_descent(b,m,points, learning_rate, num_iteration):
    for i in range(0, len(points)):
       b,m = gradient_descend_step(b,m, points, learning_rate)

    return [b,m]

def gradient_descend_step(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(0, N):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def run():
    print("allo")
    points = np.genfromtxt("./linear_regression_live_data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0

    x_values = points[:,0]
    y_values = points[:,1]
    print("Initial value:  b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))

    b,m = gradient_descent(initial_b, initial_m, points, learning_rate, 1000)

    print("After GD: b = {0}, m = {1}, error = {2}".format(b,m,compute_error(b,m,points)))
    plt.scatter(x_values, y_values)

    y_predicted = []
    for x in x_values:
        y_predicted.append(m*x + b)

    plt.plot(x_values, y_predicted)
    plt.show()


if __name__ == '__main__':
    run()