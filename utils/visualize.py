import matplotlib.pyplot as plt


def plot_results(predictions, targets):
    plt.figure(figsize=(20, 10))
    plt.plot(predictions.cpu(), label='predictions')
    plt.plot(targets.cpu(), label='targets')
    plt.legend(loc='best')
    plt.show()
