import shap
import xgboost
from sklearn import metrics
import matplotlib.pyplot as plt

def shap_explain(train_feat_df, model: xgboost.XGBClassifier):
    # Explain the model's predictions using SHAP by computing SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(train_feat_df)

    #Convert the shap values for each class to a single list
    shap_as_list=[]
    for i in range(6):
        shap_as_list.append(shap_values[:,:,i])

    # Plot the SHAP values
    shap.summary_plot(shap_as_list, train_feat_df, plot_type="bar")
    return

def confusion_matrix(data, model: xgboost.XGBClassifier):
    train_feat_df, y_train, test_clean_feat_df, y_test_clean, test_noisy_feat_df, y_test_noisy = data
    plt.figure()
    confusion_matrix_clean = metrics.confusion_matrix(y_test_clean, model.predict(test_clean_feat_df))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_clean, display_labels = ['ROC','LES','DCB','PRV','VLD', 'DTA'])
    cm_display.plot()
    plt.show()
    plt.figure()
    confusion_matrix_noisy = metrics.confusion_matrix(y_test_noisy, model.predict(test_noisy_feat_df))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_noisy, display_labels = ['ROC','LES','DCB','PRV','VLD', 'DTA'])
    cm_display.plot()
    plt.show()
    return