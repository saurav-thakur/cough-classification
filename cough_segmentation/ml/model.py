import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics

from cough_segmentation.exception import CoughSegmentationException
from cough_segmentation.logger import logging
from cough_segmentation.utils.sono_cross_val import CrossValSplit

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt


def cough_detection_model(df_from_save,result_df):

    # try:
    #     """
    #     Creates a CNN model for cough detection.

    #     Args:
    #         input_shape: A tuple representing the shape of the input data (e.g., (64, 16, 1)).

    #     Returns:
    #         A compiled Keras model.
    #     """

    #     # Input layer
    #     inputs = keras.Input(shape=input_shape)

    #     # First convolutional layer
    #     x = layers.Conv2D(16, kernel_size=(9, 3), activation="relu")(inputs)
    #     x = layers.MaxPooling2D(pool_size=(2, 1))(x)

    #     # Second convolutional layer
    #     x = layers.Conv2D(16, kernel_size=(5, 3), activation="relu")(x)
    #     x = layers.MaxPooling2D(pool_size=(2, 1))(x)

    #     # Flatten the output of convolutional layers
    #     x = layers.Flatten()(x)

    #     # First fully-connected layer with dropout
    #     x = layers.Dense(256, activation="relu")(x)
    #     x = layers.Dropout(0.5)(x)  # Dropout with probability 0.5

    #     # Second fully-connected layer with dropout
    #     x = layers.Dense(256, activation="relu")(x)
    #     x = layers.Dropout(0.5)(x)  # Dropout with probability 0.5

    #     # Output layer with softmax activation
    #     outputs = layers.Dense(1, activation="sigmoid")(x)

    #     # Create the model
    #     model = keras.Model(inputs=inputs, outputs=outputs)

    #     # Compile the model
    #     optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    #     model.compile(loss="binary_crossentropy", optimizer=optimizer,
    #         metrics=[
    #                 "accuracy",
    #                 metrics.Precision(name='precision'),
    #                 metrics.Recall(name='recall'),
    #                 metrics.AUC(name='auc_roc', curve='ROC'),
    #                 metrics.AUC(name='auc_pr', curve='PR'),
    #                 metrics.TruePositives(name='true_positives'),
    #                 metrics.FalsePositives(name='false_positives'),
    #                 metrics.TrueNegatives(name='true_negatives'),
    #                 metrics.FalseNegatives(name='false_negatives'),
    #                 # metrics.FBetaScore(name='f1_score', beta=1.0)
    #     ])

    #     logging.info("Summary of model: ")
    #     logging.info(model.summary())

    #     return model
    
    # except Exception as e:
    #     raise CoughSegmentationException(e,sys)

    try:


        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_roc_scores = []
        auc_pr_scores = []
        confusion_m = []

        mod = 'logistic'
        mod = 'xgboost'
        mod = 'svc'
        mod = 'randomforest'
        mod = 'lgbm'


        gs = True

        # Define the parameter grid for Logistic Regression
        param_grid = {}
        # Define the parameter grids for each model
        param_grids = {
            "logistic": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'sag', 'saga'],
                'max_iter': [100, 200, 300, 400, 500],
                'penalty': ["elasticnet"],
                'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0],
                # 'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                # 'random_state': [42, 100, None],
                'tol': [1e-4, 1e-3, 1e-2],
            },
            "svc": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            "xgboost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "randomforest": {
                'n_estimators': [50, 100, 200],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, None]
            },
            "lgbm": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }

        cv_strat = CrossValSplit(df_single_frame=result_df, non_cough_keys=df_from_save[df_from_save['cough_start_end'].isna()].index)
        cv_strat_data = cv_strat.cross_val(stratified=True, shuffle=True, plot=False, show_fold_info=False)
        models = {
        # "logistic": LogisticRegression(),
        # "svc": SVC(),
        "xgboost": XGBClassifier(),
        "randomforest": RandomForestClassifier(),
        "lgbm": LGBMClassifier()
        }
        best_model = None
        best_f1_mean = 0
        model_and_f1 = []



        for mod_name, model in models.items():
            
            for tuning in [False, True]:
                logging.info(f"{'With' if tuning else 'Without'} Hyperparameter Tuning:")
                
                logging.info(f"{'With' if tuning else 'Without'} Hyperparameter Tuning for {mod_name}")
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                auc_roc_scores = []
                confusion_matrices = []
                count = 1

                for fold in cv_strat_data:
                    logging.info(f"{count} fold for {mod_name}")
                    count += 1
                    X_train = result_df.loc[fold[0]].drop(columns=['amp', 'sf', 'label', 'start', 'end', 'max_amp', 'frame_index', 'key'])
                    X_test = result_df.loc[fold[1]].drop(columns=['amp', 'sf', 'label', 'start', 'end', 'max_amp', 'frame_index', 'key'])
                    y_train = result_df.loc[fold[0]]['label']
                    y_test = result_df.loc[fold[1]]['label']

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    if tuning:
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[mod_name], cv=2, scoring='roc_auc', n_jobs=-1, verbose=1)
                        grid_search.fit(X_train_scaled, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    else:
                        best_model = model
                        best_model.fit(X_train_scaled, y_train)
                        best_params = "Default parameters"

                    y_pred = best_model.predict(X_test_scaled)
                    y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_prob)
                    conf_matrix = confusion_matrix(y_test, y_pred)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    auc_roc_scores.append(roc_auc)
                    confusion_matrices.append(conf_matrix)

                mean_accuracy = np.mean(accuracy_scores)
                mean_precision = np.mean(precision_scores)
                mean_recall = np.mean(recall_scores)
                mean_f1 = np.mean(f1_scores)
                mean_auc_roc = np.mean(auc_roc_scores)
                model_and_f1.append((model,mean_f1,scaler))

                logging.info(f"Model: {mod_name} - {'Tuned' if tuning else 'Default'}")
                logging.info(f"Best Parameters: {best_params}")
                logging.info(f"Mean Accuracy: {mean_accuracy:.4f}")
                logging.info(f"Mean Precision: {mean_precision:.4f}")
                logging.info(f"Mean Recall: {mean_recall:.4f}")
                logging.info(f"Mean F1: {mean_f1:.4f}")
                logging.info(f"Mean ROC AUC: {mean_auc_roc:.4f}")
                logging.info("---")

            # if mean_f1 > best_f1_mean:
            #     best_f1_mean = mean_f1
            #     best_model = model if not tuning else best_model
            #     logging.info(f"New best model: {mod_name} ({'Tuned' if tuning else 'Default'})")
            #     logging.info(f"Best F1 score: {best_f1_mean:.4f}")

        best_model, max_f1,transformer = max(model_and_f1, key=lambda x: x[1])


        return best_model,np.mean(accuracy_scores),np.mean(precision_scores),np.mean(recall_scores),max_f1,np.mean(auc_roc_scores),confusion_m,transformer

            # # Plot confusion matrix
            # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            # disp.plot()
            # plt.show()


            # # Plot ROC Curve
            # fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            # plt.figure()
            # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            # plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc="lower right")
            # plt.show()

    except Exception as e:
        raise CoughSegmentationException(e,sys)

