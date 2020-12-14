from sklearn.model_selection import cross_val_score

def cross_validation(df, colonne, model, model_name):
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    scores = cross_val_score(model, X, y, cv=5)

    print("\nAccuracy for " + model_name + ": %0.2f (+/- %0.2f)\n\n" % (scores.mean(), scores.std() * 2))