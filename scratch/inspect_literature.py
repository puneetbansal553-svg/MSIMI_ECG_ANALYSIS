import pandas as pd

df_lit = pd.read_csv('COMPARISON/10_LITERATURE_COMPARISON/master_literature_comparison_table.csv')
print(f"Total literature entries: {len(df_lit)}")
for idx, r in df_lit.iterrows():
    print(f"{idx+1}. [{r['Publication_Year']}] {r['Authors']} - {r['Paper_Title']} ({r['Journal']}, DOI: {r['DOI']})")
    print(f"   Modality: {r['Modality_Type']} | Population: {r['Study_Population']} | N = {r['Sample_Size_N']}")
    print(f"   Comparability: {r['Comparability_Notes'][:100]}...\n")

