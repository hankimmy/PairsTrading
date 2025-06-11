import os
import pandas as pd

SIGNAL_FOLDER = "./signals"

def interpret_signal_change(prev, curr):
    if prev == curr:
        return None
    if curr == 1:
        return "LONG PX / SHORT PY"
    elif curr == -1:
        return "SHORT PX / LONG PY"
    elif curr == 0 and prev == 1:
        return "CLOSE LONG PX / SHORT PY"
    elif curr == 0 and prev == -1:
        return "CLOSE SHORT PX / LONG PY"
    else:
        return "UNKNOWN CHANGE"

def main():
    for filename in os.listdir(SIGNAL_FOLDER):
        if not filename.endswith(".csv"):
            continue

        pair_path = os.path.join(SIGNAL_FOLDER, filename)
        df = pd.read_csv(pair_path, parse_dates=["date"])
        df = df.sort_values("date")

        if len(df) < 2:
            continue

        prev_row = df.iloc[-2]
        curr_row = df.iloc[-1]
        prev_signal = prev_row['signal']
        curr_signal = curr_row['signal']

        if prev_signal == curr_signal:
            continue

        instruction = interpret_signal_change(prev_signal, curr_signal)
        if instruction:
            ticker_x, ticker_y = filename.replace(".csv", "").split("_")
            px = curr_row['px']
            py = curr_row['py']
            print(f"[TRADE] {instruction}: {ticker_x} at ${px:.2f}, {ticker_y} at ${py:.2f} (Signal: {prev_signal} → {curr_signal})")

if __name__ == "__main__":
    main()
