import os
import pandas as pd

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

def process_signal_file(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return

    df = df.sort_values("date")

    if len(df) < 2:
        return

    prev_row = df.iloc[-2]
    curr_row = df.iloc[-1]
    prev_signal = prev_row['signal']
    curr_signal = curr_row['signal']

    if prev_signal == curr_signal:
        return

    instruction = interpret_signal_change(prev_signal, curr_signal)
    if instruction:
        filename = os.path.basename(filepath)
        ticker_x, ticker_y = filename.replace(".csv", "").split("_")
        px = curr_row['px']
        py = curr_row['py']
        print(f"[TRADE] {instruction}: {ticker_x} at ${px:.2f}, {ticker_y} at ${py:.2f} (Signal: {prev_signal} → {curr_signal})")

def main():
    for root, _, files in os.walk("."):
        if not root.endswith("_signals"):
            continue
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                process_signal_file(filepath)

if __name__ == "__main__":
    main()
