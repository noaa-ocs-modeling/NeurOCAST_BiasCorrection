import os
import re
import sys

def patch_fno():
    try:
        import physicsnemo
        base_dir = os.path.dirname(physicsnemo.__file__)
        fno_path = os.path.join(base_dir, "models", "fno", "fno.py")

        if not os.path.exists(fno_path):
            print(f"fno.py not found at {fno_path}")
            sys.exit(1)

        with open(fno_path, "r") as f:
            lines = f.readlines()

        in_fno1d = False
        patched = False
        for i, line in enumerate(lines):
            # Detect start of FNO1DEncoder
            if re.match(r'^\s*class\s+FNO1DEncoder\(.*\):', line):
                in_fno1d = True
                continue

            if in_fno1d:
                # Exit if we reach a new class
                if re.match(r'^\s*class\s+\w+\(.*\):', line):
                    break

                # Find the target line to patch
                if "x = conv(x) + w(x)" in line and 'x.to(torch.float32)' not in lines[i - 1]:
                    indent = re.match(r'^(\s*)', line).group(1)
                    lines.insert(i, f"{indent}x = x.to(torch.float32)\n")
                    patched = True
                    break

        if patched:
            with open(fno_path, "w") as f:
                f.writelines(lines)
            print("Patched fno.py successfully.")
        else:
            print("Patch already applied or target line not found.")

    except Exception as e:
        print(f"Error during patch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    patch_fno()
