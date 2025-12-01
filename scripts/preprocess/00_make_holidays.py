import pandas as pd, holidays, yaml, os
cfg = yaml.safe_load(open("configs/default.yaml"))
years = range(2017, 2023)
at = holidays.country_holidays("AT", years=years)
rows = [{"date": pd.Timestamp(d).tz_localize("UTC"), "holiday_name": name}
        for d, name in sorted(at.items())]
out = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
os.makedirs(cfg["data_path"], exist_ok=True)
out.to_csv(cfg["holidays_csv"], index=False)
print(f"Saved {cfg['holidays_csv']} with {len(out)} rows")
