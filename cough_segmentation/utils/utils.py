import pickle



def save_pickle_model(model,filename):
    # Save the model to disk
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_pickle_model(filename):
    # Save the model to disk
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

    