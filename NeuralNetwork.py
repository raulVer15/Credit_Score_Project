import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
from datetime import datetime
import os
#from DataLogging import load_data
from DataCleaning import clean_data

def train_NN(df):
        # Define features and target
        continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                            'Delay_from_due_date', 'Num_of_Delayed_Payment',
                            'Changed_Credit_Limit', 'Outstanding_Debt',
                            'Credit_Utilization_Ratio', 'Credit_History_Age',
                            'Total_EMI_per_month', 'Amount_invested_monthly',
                            'Monthly_Balance', 'Num_Bank_Accounts', 'Num_Credit_Inquiries']

        categorical_features = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 
                                'Payment_of_Min_Amount', 'Last_Loan_1', 'Last_Loan_2', 
                                'Last_Loan_3', 'Last_Loan_4', 'Last_Loan_5']

        target = ['Credit_Score']
        # Validate columns
        missing_columns = [col for col in continuous_features + categorical_features + target if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing in the dataset: {missing_columns}") 
        
        encoder = OneHotEncoder(handle_unknown="ignore")
        
        # Encode categorical features
        encoded_features = encoder.fit_transform(df[categorical_features])
        
        #Calculating column len
        num_continuous_features = len(continuous_features)
        num_encoded_columns = encoded_features.shape[1]
        total_columns = num_continuous_features + num_encoded_columns
        
        # Load data
        scaler = StandardScaler()
        scaled_continuous = scaler.fit_transform(df[continuous_features])
        
        
        encoded_target = encoder.fit_transform(df[target])
        encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))
        df = pd.concat([df, encoded_target_df], axis=1)


        X = np.hstack([scaled_continuous, encoded_features.toarray()])
        y = encoded_target.toarray()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Get input dimensions
        total_columns = X_train.shape[1]

        # Build model
        model = Sequential()
        model.add(Dense(total_columns, input_dim=total_columns, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(total_columns * 2, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(total_columns * 4, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(total_columns * 2, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        print("Training model")
        # Train the model
        model.fit(X_train, y_train,
                validation_split=0.2,
                epochs=80,
                batch_size=128,
                callbacks=[early_stopping, lr_scheduler])

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print metrics
        print("Train NN:")
        print("**********")
        print(f"[{current_time}] Model Accuracy: {test_acc}")
        print(f"[{current_time}] Model Precision: {precision}")
        print(f"[{current_time}] Model Recall: {recall}")
        print(f"[{current_time}] Model f1_score: {f1}")
        print(f"[{current_time}] Model Confusion Matrix: \n{conf_matrix}")

        # Return metrics in an array
        return [test_acc, precision, recall, f1, conf_matrix]

    # except Exception as e:
    #     # Handle any errors that occur
    #     return f"An error occurred: {str(e)}"

    # finally:
    #     # Optional: Cleanup code or final statement
    #     print("Model training and evaluation process completed.")
        
# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir))
file_path = os.path.join(parent_dir, "data", "Credit_score_data.csv")

#train_NN(clean_data(load_data(file_path)))
#train_NN(load_data(file_path))





