import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_roc(true, score):
    preds = score[:,1]
    fpr, tpr, threshold = metrics.roc_curve(true, preds)
    roc_auc = metrics.auc(fpr, tpr)
 
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./ROC.png')

def main():
    y_score = np.loadtxt("./y_score.txt",delimiter=",")
    y_pred = np.loadtxt("./y_pred.txt",delimiter=",")
    y_true = np.loadtxt("./y_true.txt",delimiter=",")
    plot_roc(y_true, y_score)

    print("Average_precision_score : " + str(metrics.average_precision_score(y_true, y_score[:,1])))
    print("Roc_auc_score : " + str(metrics.roc_auc_score(y_true, y_score[:,1])))
    print("Recall : " + str(metrics.recall_score(y_true, y_pred)))
    print("Accuracy : " + str(metrics.accuracy_score(y_true, y_pred)))

if __name__ == '__main__':
    main()

