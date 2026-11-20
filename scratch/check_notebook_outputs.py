import json

nb = json.load(open('notebooks/05_rpeak_detection.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        outs = c.get('outputs', [])
        print(f"Code cell index {i}: {len(outs)} outputs")
        for o in outs:
            if 'text' in o:
                txt = ''.join(o['text'])
                lines = txt.strip().splitlines()
                if len(lines) > 5:
                    print(f"   First 2 lines: {lines[:2]}")
                    print(f"   Last 2 lines: {lines[-2:]}")
                else:
                    print(f"   All lines: {lines}")

