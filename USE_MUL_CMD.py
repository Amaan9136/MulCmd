from mulcmds.mul_cmd import MulCmd

def run_mulcmd_workflow():
    # Initialize the class
    mul_cmd = MulCmd()

    # Load the data file
    filenames = ["iris.csv"]  # Replace this with your actual file(s)
    print("Loading file(s)...")
    print(mul_cmd.file(filenames))

    # Encode the features (modify this based on your actual dataset and command)
    print("\nEncoding features...")
    encoding_command = 'label=0,1 onehot=2,3'  # Adjust column indices as needed
    print(mul_cmd.encode_features(encoding_command))

    # Set the feature range
    print("\nSetting feature range...")
    start_feature_index = 0  # Replace with the start index of features
    end_feature_index = 2    # Replace with the end index of features
    print(mul_cmd.features_range(start_feature_index, end_feature_index))

    # Set the target range
    print("\nSetting target range...")
    target_columns = [3]  # Replace with the target column index
    print(mul_cmd.target_range(*target_columns))

    # Split the data into train and test sets
    print("\nSplitting data...")
    split_ratio = 0.2  # 20% for testing, 80% for training
    print(mul_cmd.split(split_ratio))

    # Set the machine learning model (change model name to a valid one, e.g., 'linear_regression', 'decision_tree')
    model_name = 'linear_regression'  # Replace with your desired model
    print("\nSetting the model...")
    print(mul_cmd.set_model(model_name))

    # Train the model
    print("\nTraining the model...")
    print(mul_cmd.train())

    # Make predictions
    print("\nMaking predictions...")
    print(mul_cmd.print_predict())

    # Print the accuracy of the model
    print("\nPrinting accuracy...")
    print(mul_cmd.print_accuracy())

    # Print the R2 score (for regression tasks)
    print("\nPrinting R2 score...")
    print(mul_cmd.print_r2())

    # Plot the data (optional)
    print("\nPlotting data...")
    plot_data_file, plot_data_message = mul_cmd.plot_data()
    print(plot_data_message)
    print(f"Plot saved to: {plot_data_file}")

    # Plot the predicted data (optional)
    print("\nPlotting predictions...")
    plot_predict_file, plot_predict_message = mul_cmd.plot_predict()
    print(plot_predict_message)
    print(f"Prediction plot saved to: {plot_predict_file}")

if __name__ == "__main__":
    run_mulcmd_workflow()
