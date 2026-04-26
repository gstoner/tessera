
# Placeholder: generate a simple CSV with stage timings and flops/bytes
import csv, sys
with open(sys.argv[1] if len(sys.argv)>1 else "roofline.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stage","flops","bytes","time_ms"])
    w.writerow(["radix4_stage0", 1.0e9, 2.0e9, 0.8])
    w.writerow(["transpose",     0.1e9, 8.0e9, 1.2])
