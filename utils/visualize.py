import matplotlib.pyplot as plt


def plot_results(predictions, targets, title='Results'):
    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.plot(predictions.cpu(), label='predictions')
    plt.plot(targets.cpu(), label='targets')
    plt.legend(loc='best')
    plt.show()
