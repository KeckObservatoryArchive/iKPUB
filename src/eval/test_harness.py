from data_pipes.preprocess import load_pubs
    

if __name__ == "__main__":
    
    model = ...
    
    # Load data
    pubs = load_pubs()
    X_train, X_test, y_train, y_test = train_test_split(pubs[[...]], pubs[[...]])
    
    # Optionally train
    model.train(X_train, X_test)
    
    # Evaluate
    predictions = model.predict(X_test)
    
    # Print summary statistics

    