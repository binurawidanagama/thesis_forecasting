from src.train.train_encoder import run
from src.config import load_config
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args=ap.parse_args(); run(args.config)

