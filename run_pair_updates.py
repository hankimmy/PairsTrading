import datetime
import holidays
import subprocess

def is_trading_day(date):
    us_holidays = holidays.US()
    return date.weekday() < 5 and date not in us_holidays  # Mon–Fri and not a holiday

if __name__ == "__main__":
    today = datetime.date.today()
    if is_trading_day(today):
        project_dir = "/Users/hankim/PycharmProjects/PairTrading"
        python_bin = f"{project_dir}/venv/bin/python3"

        subprocess.run([python_bin, "update_best_pairs.py"], cwd=project_dir)
        subprocess.run([python_bin, "find_trades.py"], cwd=project_dir)
