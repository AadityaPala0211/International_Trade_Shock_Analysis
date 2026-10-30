"""Companion-workbook metadata and conservative OECD ICIO provenance audit."""
from pathlib import Path
from openpyxl import load_workbook

OECD_ICIO_DATASET_URL = "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html"


def inspect_icio_readme(path, preview_rows=30, preview_columns=12):
    """Return workbook properties, sheet dimensions and a bounded cell preview."""
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "file_not_found"}
    try:
        wb = load_workbook(path, read_only=True, data_only=True)
        props = wb.properties
        sheets = []
        for ws in wb.worksheets:
            preview = []
            for row in ws.iter_rows(min_row=1, max_row=preview_rows, min_col=1,
                                    max_col=preview_columns, values_only=True):
                values = [None if value is None else str(value) for value in row]
                if any(value is not None for value in values):
                    preview.append(values)
            sheets.append({"name": ws.title, "max_row": ws.max_row,
                           "max_column": ws.max_column, "preview": preview})
        result = {
            "available": True, "path": str(path), "sheet_names": wb.sheetnames,
            "properties": {
                "title": props.title, "subject": props.subject, "creator": props.creator,
                "description": props.description,
                "created": str(props.created) if props.created else None,
                "modified": str(props.modified) if props.modified else None,
            },
            "sheets": sheets,
        }
        wb.close()
        return result
    except Exception as exc:
        return {"available": True, "path": str(path),
                "inspection_error": f"{type(exc).__name__}: {exc}"}


def summarize_icio_provenance(workbook_info, loaded_csvs):
    """Separate official release context from what the local files actually prove."""
    text_parts = []
    if workbook_info.get("available") and not workbook_info.get("inspection_error"):
        for sheet in workbook_info.get("sheets", []):
            text_parts.append(sheet.get("name", ""))
            for row in sheet.get("preview", []):
                text_parts.extend("" if value is None else str(value) for value in row)
        text_parts.extend(str(v) for v in workbook_info.get("properties", {}).values() if v is not None)
    companion_text = " ".join(text_parts)
    lower = companion_text.lower()
    mentions_icio = "icio" in lower or "inter-country input-output" in lower
    mentions_2025 = "2025" in lower
    if mentions_icio and mentions_2025:
        companion_status = "companion_workbook_mentions_oecd_icio_2025"
    elif mentions_2025:
        companion_status = "companion_workbook_mentions_2025_but_not_explicitly_icio"
    else:
        companion_status = "companion_workbook_does_not_establish_2025_icio_edition"

    files = []
    for item in loaded_csvs:
        files.append({
            "path": item.get("source_path"),
            "sha256": item.get("source_sha256"),
            "shape": list(item.get("Z").shape) if item.get("Z") is not None else None,
            "node_count": item.get("n_nodes"),
        })
    return {
        "local_companion_workbook_status": companion_status,
        "companion_workbook_mentions_icio": bool(mentions_icio),
        "companion_workbook_mentions_2025": bool(mentions_2025),
        "csv_file_checksums_and_dimensions": files,
        "what_is_verified_locally": (
            "The selected CSV files were parsed, their dimensions and SHA-256 checksums recorded, "
            "and their detailed node universes were aligned."
        ),
        "what_is_not_independently_proven": (
            "A companion workbook preview cannot by itself prove that each individual CSV was "
            "downloaded from the same OECD release. Retain the original OECD download page/archive "
            "or source manifest as file-level provenance."
        ),
        "official_oecd_reference": OECD_ICIO_DATASET_URL,
        "official_reference_context": (
            "The OECD ICIO dataset page lists a 2025 regular ICIO release covering 1995-2022. "
            "This supports the plausibility of using 2019 and 2022 from that release, but does not "
            "independently identify the provenance of these local CSV files."
        ),
    }
