import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _visualize_performance(ds, acc, y_pred, ax=None, title=None):
     if ax is None:
         _, ax = plt.subplots(1, 1, figsize=(8, 8))

     df = pd.DataFrame(confusion_matrix(np.squeeze(ds.y), y_pred),
                       index=ds.labels,
                       columns=ds.labels)

     hm = sns.heatmap(
         df, cbar=False, annot=True, fmt='d', cmap='Blues', ax=ax)

     xlabels = hm.xaxis.get_ticklabels()
     hm.xaxis.set_ticklabels(xlabels, rotation=45, ha='right')

     if title is not None:
         fmt = title + ", Total Accuracy is {:.3f}"
     else:
         fmt = "Total Accuracy is {:.3f}"

     ax.set_title(fmt.format(acc))

     plt.tight_layout()
