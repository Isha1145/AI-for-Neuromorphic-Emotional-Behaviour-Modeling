import pickle, os

def load_subject(sid, data_dir="data"):
    with open(os.path.join(data_dir, f"s{sid:02d}.dat"), "rb") as f:
        return pickle.load(f, encoding="latin1")
