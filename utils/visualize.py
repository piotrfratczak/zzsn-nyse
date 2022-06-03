import matplotlib.pyplot as plt


def plot_results(predictions, targets):
    plt.figure(figsize=(20, 10))
    plt.plot(predictions, label='predictions')
    plt.plot(targets, label='targets')
    plt.legend(loc='best')
    plt.show()
