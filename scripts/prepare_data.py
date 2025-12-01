from src.config import load_config
from src.dataio.preprocess import build_master

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a=ap.parse_args()
    build_master(load_config(a.config))
    print("Prepared interim datasets.")

