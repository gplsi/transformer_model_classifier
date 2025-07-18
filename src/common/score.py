import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


plt.style.use('ggplot')



def scorePredict(y_test, y_hat, labels):
    from sklearn.metrics import f1_score

    matriz = confusion_matrix(y_test, y_hat, labels=labels)
    f1_score_value = f1_score(y_test, y_hat, average='macro')
    value_class = classification_report(y_test, y_hat, digits=5)
    result = 'Matriz de Confusión:' +'\n' + str(matriz) + '\n' + value_class+ '\n' + str(f1_score_value)
    class_names = labels
    plot_confusion_matrix(matriz, class_names, normalize=False, )
    return result, f1_score_value


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    # ,vmin=-3000, vmax=5000
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(ticks=range(6), spacing='uniform', anchor=(1.0, 0.0))
    # ax, _ = plt.colorbar.make_axes(plt.gca(), shrink=0.5)
    # cbar = plt.colorbar.ColorbarBase(ax, cmap=cm,
    #                    norm=plt.colors.Normalize(vmin=-0.5, vmax=1.5))
    # cbar.set_clim(-2.0, 2.0)
    # plt.get_colorbar().set_clim(0,1)
    # plt.colorbar(cbar)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=100)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('matrix.png')
    plt.show()
