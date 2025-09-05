#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("csv")
  ap.add_argument("--x", default=None)
  ap.add_argument("--y", default="gflops")
  args = ap.parse_args()
  df = pd.read_csv(args.csv)
  if args.x and args.x in df.columns:
    xs = sorted(df[args.x].unique())
    ys = [df[df[args.x]==x][args.y].mean() for x in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(args.x); plt.ylabel(args.y); plt.title(f"{args.y} vs {args.x}")
    plt.tight_layout(); plt.show()
  else:
    print(df.describe())

if __name__ == "__main__":
  main()
