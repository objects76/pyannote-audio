"""Compare pyan30 and pyan40 weight files."""

import torch

WEIGHTS = {
    "pyan30": "/home/jjkim/Desktop/work/Diarization/rsup-gitlab/weight-for-spk-diar/pyan30/pytorch_model.bin",
    "pyan40": "/home/jjkim/Desktop/work/Diarization/wsdiar_ws3/src/spkdiar/weights/pyan40/pytorch_model.bin",
}

ckpt30 = torch.load(WEIGHTS["pyan30"], map_location="cpu", weights_only=False)
ckpt40 = torch.load(WEIGHTS["pyan40"], map_location="cpu", weights_only=False)

sd30 = ckpt30["state_dict"]
sd40 = ckpt40["state_dict"]

keys30 = set(sd30.keys())
keys40 = set(sd40.keys())

only30 = keys30 - keys40
only40 = keys40 - keys30
common = keys30 & keys40

print(f"pyan30 keys: {len(keys30)}")
print(f"pyan40 keys: {len(keys40)}")
print(f"common keys: {len(common)}")
if only30:
    print(f"only in pyan30: {sorted(only30)}")
if only40:
    print(f"only in pyan40: {sorted(only40)}")

all_equal = True
diff_keys = []
for key in sorted(common):
    if not torch.equal(sd30[key], sd40[key]):
        all_equal = False
        max_diff = (sd30[key] - sd40[key]).abs().max().item()
        diff_keys.append((key, sd30[key].shape, max_diff))

print()
if all_equal and not only30 and not only40:
    print("RESULT: weights are IDENTICAL")
else:
    if diff_keys:
        print(f"RESULT: {len(diff_keys)} keys have different values:")
        for key, shape, max_diff in diff_keys:
            print(f"  {key} {list(shape)}  max_diff={max_diff:.6e}")
    elif only30 or only40:
        print("RESULT: different keys but common values are identical")
    else:
        print("RESULT: common keys are identical")
