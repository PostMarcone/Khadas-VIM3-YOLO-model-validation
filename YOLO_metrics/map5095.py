import subprocess
import argparse

thresholds = [x / 100 for x in range(50, 100, 5)]
#thresholds = [0.9, 0.95]
maps = []

parser = argparse.ArgumentParser()
parser.add_argument("--dr", required=True, help="Relative path to the detection results")
args = parser.parse_args()

model = args.dr

for t in thresholds:
    print(f"Running evaluation at IoU {t}")
    result = subprocess.run(
        ["python", "main.py", "--minoverlap", str(t), "--mdl", model],
        capture_output=True, text=True
    )

    for line in result.stdout.splitlines():
        if "mAP" in line:
            try:
                last_part = line.strip().split()[-1]
                maps.append(float(last_part.strip('%')))
            except:
                pass

if maps:
    print("All mAPs:", maps)
    print("mAP@50:95 =", sum(maps) / len(maps))
else:
    print("No mAP values found")
