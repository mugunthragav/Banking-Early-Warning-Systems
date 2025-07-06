import pickle
with open('models/all_commodities_model.pkl', 'rb') as f:
    results = pickle.load(f)
print(len(results), list(results.keys()))