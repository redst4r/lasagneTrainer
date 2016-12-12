import numpy as np

"""
machine learning metrics
"""
def get_auc(scores, truelabel, do_plot=False):
    from sklearn.metrics import roc_curve, auc, accuracy_score
    fpr, tpr, thresholds = roc_curve(truelabel, scores, drop_intermediate=False)
    the_auc = auc(fpr, tpr)
    if do_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(fpr, tpr, c=thresholds)
        plt.title("AUC %.02f" % the_auc)
        plt.plot(fpr,tpr,'k')
        plt.xlim([0,1])
        plt.ylim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
    return the_auc

def acc_vs_cutoff(y,scores, do_plot=False):
    from sklearn.metrics import accuracy_score

    acc= []
    cutoff_range = np.linspace(0,1,100)
    for cutoff in cutoff_range:
        y_hat = scores > cutoff
        a = accuracy_score(y, y_hat)
        acc.append(a)

    acc = np.array(acc)

    if do_plot:
        import matplotlib.pyplot as plt

        plt.plot(cutoff_range, acc)
        plt.xlabel('prob. cutoff')
        plt.ylabel('Accuracy')
    return acc