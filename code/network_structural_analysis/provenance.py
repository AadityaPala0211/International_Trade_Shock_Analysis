"""Lightweight provenance extraction from the OECD ICIO companion workbook."""
from pathlib import Path
from openpyxl import load_workbook

def inspect_icio_readme(path,preview_rows=30,preview_columns=12):
    """Return workbook properties, sheet dimensions and a bounded preview.

    The workbook is provenance documentation, not transaction data. Inspection
    errors are recorded rather than preventing the CSV network workflow.
    """
    path=Path(path)
    if not path.exists():
        return {"available":False,"path":str(path),"reason":"file_not_found"}
    try:
        wb=load_workbook(path,read_only=True,data_only=True); props=wb.properties
        sheets=[]
        for ws in wb.worksheets:
            preview=[]
            for row in ws.iter_rows(min_row=1,max_row=preview_rows,min_col=1,
                                    max_col=preview_columns,values_only=True):
                values=[None if value is None else str(value) for value in row]
                if any(value is not None for value in values): preview.append(values)
            sheets.append({"name":ws.title,"max_row":ws.max_row,
                           "max_column":ws.max_column,"preview":preview})
        result={"available":True,"path":str(path),"sheet_names":wb.sheetnames,
            "properties":{"title":props.title,"subject":props.subject,"creator":props.creator,
              "description":props.description,"created":str(props.created) if props.created else None,
              "modified":str(props.modified) if props.modified else None},"sheets":sheets}
        wb.close(); return result
    except Exception as exc:
        return {"available":True,"path":str(path),"inspection_error":f"{type(exc).__name__}: {exc}"}
