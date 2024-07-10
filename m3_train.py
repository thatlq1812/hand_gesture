from m3_module import *

if __name__ == "__main__":
    trainer = RandomForestTrainer(csv_file='m3_info.csv')
    trainer.preprocess_data(0.0) # Corr < 0.0 will be removed
    trainer.split_data(test_size=0.2, random_state=42)
    trainer.create_model(
        ip_n_estimators=200,
        ip_max_depth=30,
        ip_min_samples_split=5,
        ip_min_samples_leaf=2,
        ip_max_features='sqrt',
        ip_bootstrap=True,
        ip_random_state=42
    )
    trainer.train_model(save_location='m3_model.joblib')
    trainer.evaluate_model()