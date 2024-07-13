from m3_module import *

if __name__ == "__main__":
    trainer = RandomForestTrainer('m3_info.csv') # Parameters is the dataset file
    trainer.preprocess_data() # Preprocess the data
    trainer.split_data(test_size=0.2, random_state=42) # Split the data
    
    # Create the model
    trainer.create_model(
        ip_n_estimators=200,
        ip_max_depth=30,
        ip_min_samples_split=5,
        ip_min_samples_leaf=2,
        ip_max_features='sqrt',
        ip_bootstrap=True,
        ip_random_state=42
    )

    # Save and evaluate the model
    trainer.train_model('m3_model.joblib') # Parameters is saving location for the model
    trainer.evaluate_model() # Evaluate the model