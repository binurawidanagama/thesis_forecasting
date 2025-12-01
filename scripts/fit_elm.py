from src.train.fit_elm import run
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args=ap.parse_args(); run(args.config)

