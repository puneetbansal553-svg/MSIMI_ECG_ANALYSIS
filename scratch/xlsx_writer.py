import zipfile
import io
import html
import pandas as pd

def write_dataframes_to_xlsx(dfs_dict, output_path):
    """
    Pure Python multi-sheet xlsx writer using standard library zipfile.
    dfs_dict: dict of {sheet_name: pd.DataFrame}
    output_path: Path or str
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        sheet_names = list(dfs_dict.keys())
        n_sheets = len(sheet_names)
        
        # [Content_Types].xml
        sheet_types = ''.join([f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(n_sheets)])
        content_types = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>{sheet_types}</Types>'
        z.writestr('[Content_Types].xml', content_types)
        
        # _rels/.rels
        rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
        z.writestr('_rels/.rels', rels)
        
        # xl/_rels/workbook.xml.rels
        wb_rels_items = ''.join([f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i+1}.xml"/>' for i in range(n_sheets)])
        wb_rels = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{wb_rels_items}</Relationships>'
        z.writestr('xl/_rels/workbook.xml.rels', wb_rels)
        
        # xl/workbook.xml
        wb_sheets = ''.join([f'<sheet name="{html.escape(name[:31])}" sheetId="{i+1}" r:id="rId{i+1}"/>' for i, name in enumerate(sheet_names)])
        wb = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{wb_sheets}</sheets></workbook>'
        z.writestr('xl/workbook.xml', wb)
        
        # Each worksheet
        for i, (name, df) in enumerate(dfs_dict.items()):
            rows_xml = []
            
            # Header row
            header_cols = []
            for col_val in df.columns:
                s_val = html.escape(str(col_val))
                header_cols.append(f'<c t="inlineStr"><is><t>{s_val}</t></is></c>')
            rows_xml.append(f'<row r="1">{"".join(header_cols)}</row>')
            
            # Data rows
            for r_idx, row in enumerate(df.itertuples(index=False), start=2):
                cols_xml = []
                for val in row:
                    if pd.isna(val):
                        cols_xml.append('<c/>')
                    elif isinstance(val, (int, float, bool)) and not isinstance(val, str):
                        cols_xml.append(f'<c><v>{val}</v></c>')
                    else:
                        s_val = html.escape(str(val))
                        cols_xml.append(f'<c t="inlineStr"><is><t>{s_val}</t></is></c>')
                rows_xml.append(f'<row r="{r_idx}">{"".join(cols_xml)}</row>')
                
            sheet_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
            z.writestr(f'xl/worksheets/sheet{i+1}.xml', sheet_xml)
            
    with open(output_path, 'wb') as f:
        f.write(buf.getvalue())
    print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    df1 = pd.DataFrame({'ColA': [1, 2, 3], 'ColB': ['X', 'Y', 'Z']})
    df2 = pd.DataFrame({'Num': [10.5, 20.3], 'Desc': ['Test1', 'Test2']})
    write_dataframes_to_xlsx({'Sheet1': df1, 'Sheet2': df2}, 'scratch/test_out.xlsx')

