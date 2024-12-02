import xgboost
import numpy as np
import pandas as pd

#Format input data
def train(data, model: xgboost.XGBClassifier):
    train_feat_df, y_train, test_clean_feat_df, y_test_clean, test_noisy_feat_df, y_test_noisy = data
    
    model.fit(train_feat_df,y_train)

    print("Train Clean Acc =", np.sum(y_train==model.predict(train_feat_df))/len(y_train))
    print("Test Clean Acc =", np.sum(y_test_clean==model.predict(test_clean_feat_df))/len(y_test_clean))
    print("Test Noisy Acc =", np.sum(y_test_noisy==model.predict(test_noisy_feat_df))/len(y_test_noisy))
