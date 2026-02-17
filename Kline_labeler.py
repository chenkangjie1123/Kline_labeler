import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 直接复用你已有的函数
from data_download import load_local


# -----------------------------
# Candlestick plotting (no extra deps)
# -----------------------------
def plot_candles(ax, df, title="", center_idx=None):
    ax.clear()
    ax.set_title(title, fontsize=11)

    if df.empty:
        ax.text(0.5, 0.5, "Empty window", ha="center", va="center")
        ax.set_axis_off()
        return

    x = np.arange(len(df))
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # Wicks
    ax.vlines(x, l, h, linewidth=1)

    # Bodies
    up = c >= o
    down = ~up

    body_bottom = np.where(up, o, c)
    body_top = np.where(up, c, o)
    body_h = body_top - body_bottom
    body_h = np.where(body_h == 0, 1e-9, body_h)

    ax.bar(x[up], body_h[up], bottom=body_bottom[up], width=0.6, align="center")
    ax.bar(x[down], body_h[down], bottom=body_bottom[down], width=0.6, align="center")

    if center_idx is not None and 0 <= center_idx < len(df):
        ax.axvline(center_idx, linestyle="--", linewidth=1)

    # x ticks
    n = len(df)
    ticks = np.linspace(0, n - 1, num=min(6, n), dtype=int)
    labels = [str(df.index[i])[:19] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    ax.grid(True, alpha=0.2)


# -----------------------------
# Labeler
# -----------------------------
class KlineLabeler:
    def __init__(
        self,
        df: pd.DataFrame,
        out_csv: str,
        before: int = 120,
        after: int = 120,
        step: int = 1,
        start_at: int = 0,
        labels_map: dict | None = None,
        allow_skip: bool = True,
    ):
        self.df = df.sort_index()
        self.out_csv = out_csv
        self.before = before
        self.after = after
        self.step = step
        self.allow_skip = allow_skip

        if labels_map is None:
            labels_map = {
                "1": "trend_up",
                "2": "trend_down",
                "3": "reversal_up",
                "4": "reversal_down",
                "5": "breakout",
                "6": "breakdown",
                "7": "range",
                "8": "spike",
                "9": "other",
            }
        self.labels_map = labels_map

        # load old labels
        self.labels = {}
        self.history = []
        if os.path.exists(out_csv):
            try:
                old = pd.read_csv(out_csv)
                for _, r in old.iterrows():
                    self.labels[int(r["index"])] = str(r["label"])
            except Exception as e:
                print(f"[WARN] failed to load old labels: {e}")

        self.i = int(start_at)
        self.i = max(0, min(self.i, len(self.df) - 1))

        self.fig, (self.axL, self.axR) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._update_plot()

    def _window(self, center: int):
        l = max(0, center - self.before)
        r = min(len(self.df), center + self.after + 1)
        left_df = self.df.iloc[l:center + 1]
        right_df = self.df.iloc[center:r]
        return left_df, right_df

    def _status_text(self):
        t = self.df.index[self.i]
        cur_label = self.labels.get(self.i, "")
        keys_help = " ".join([f"[{k}]={v}" for k, v in self.labels_map.items()])
        extra = f"  current_label: {cur_label}" if cur_label else "  current_label: (none)"
        help2 = "  [n]=next  [p]=prev  [g]=jump  [u]=undo  [s]=save  [q]=quit"
        if self.allow_skip:
            help2 += "  [k]=skip"
        return f"idx={self.i}/{len(self.df)-1}  time={str(t)[:19]}{extra}\n{keys_help}\n{help2}"

    def _update_plot(self):
        left_df, right_df = self._window(self.i)
        plot_candles(self.axL, left_df, title="Before (incl. center)", center_idx=len(left_df) - 1)
        plot_candles(self.axR, right_df, title="After (incl. center)", center_idx=0)
        self.fig.suptitle(self._status_text(), fontsize=10, y=0.98)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _save(self):
        rows = []
        for idx, lab in sorted(self.labels.items(), key=lambda x: x[0]):
            t = self.df.index[idx]
            rows.append({"index": idx, "time": str(t), "label": lab})
        out = pd.DataFrame(rows)
        out.to_csv(self.out_csv, index=False)
        print(f"[SAVE] {len(out)} labels -> {self.out_csv}")

    def _set_label(self, label: str):
        prev = self.labels.get(self.i, None)
        self.history.append((self.i, prev))
        self.labels[self.i] = label
        self._advance()

    def _advance(self):
        self.i = min(len(self.df) - 1, self.i + self.step)
        self._update_plot()

    def _back(self):
        self.i = max(0, self.i - self.step)
        self._update_plot()

    def _skip(self):
        prev = self.labels.get(self.i, None)
        self.history.append((self.i, prev))
        if self.i in self.labels:
            del self.labels[self.i]
        self._advance()

    def _undo(self):
        if not self.history:
            print("[UNDO] nothing to undo")
            return
        idx, prev = self.history.pop()
        if prev is None:
            if idx in self.labels:
                del self.labels[idx]
        else:
            self.labels[idx] = prev
        self.i = idx
        self._update_plot()

    def _jump(self):
        try:
            raw = input("Jump to index (int) or time prefix (e.g. 2025-12-31 23:59): ").strip()
            if raw.isdigit():
                j = int(raw)
                self.i = max(0, min(j, len(self.df) - 1))
            else:
                s = raw
                candidates = [k for k, t in enumerate(self.df.index.astype(str)) if s in t]
                if candidates:
                    self.i = candidates[0]
                else:
                    print("[JUMP] not found.")
            self._update_plot()
        except Exception as e:
            print(f"[JUMP] failed: {e}")

    def on_key(self, event):
        k = event.key
        if k in self.labels_map:
            self._set_label(self.labels_map[k])
        elif k == "n":
            self._advance()
        elif k == "p":
            self._back()
        elif k == "k":
            self._skip()
        elif k == "u":
            self._undo()
        elif k == "s":
            self._save()
        elif k == "g":
            self._jump()
        elif k == "q":
            self._save()
            plt.close(self.fig)

    def run(self):
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./binance_data")
    ap.add_argument("--market", type=str, default="futures")
    ap.add_argument("--interval", type=str, default="1m")
    ap.add_argument("--symbol", type=str, required=True)   # ✅ 直接传 ETHUSDT
    ap.add_argument("--fmt", type=str, default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--out", type=str, default="", help="output labels csv path")
    ap.add_argument("--before", type=int, default=120)
    ap.add_argument("--after", type=int, default=120)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start_at", type=int, default=0)
    ap.add_argument("--labels_json", type=str, default="")
    args = ap.parse_args()

    df = load_local(args.out_dir, args.market, args.interval, args.symbol, args.fmt)
    if df.empty:
        raise ValueError(f"Loaded empty df for {args.symbol} {args.market} {args.interval}")

    # 确保必需列存在
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}, available={df.columns.tolist()}")

    labels_map = None
    if args.labels_json:
        labels_map = json.loads(args.labels_json)

    out_csv = args.out if args.out else f"./labels_{args.symbol}_{args.interval}.csv"

    app = KlineLabeler(
        df=df,
        out_csv=out_csv,
        before=args.before,
        after=args.after,
        step=args.step,
        start_at=args.start_at,
        labels_map=labels_map,
        allow_skip=True,
    )
    app.run()


if __name__ == "__main__":
    main()
    
    
# python Kline_labeler.py --symbol ETHUSDT --interval 1m --market futures --out_dir ./binance_data --fmt parquet