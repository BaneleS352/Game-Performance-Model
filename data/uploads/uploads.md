# `data/uploads/` Folder Guide

## Folder purpose

Stores original workbooks received through the local upload API. File names are
sanitised and prefixed by a UTC upload id.

## File inventory

| File pattern | Purpose | Dependencies | Description |
|---|---|---|---|
| `{upload_id}_{safe_name}.csv` | CSV source | pandas | One logical worksheet. |
| `{upload_id}_{safe_name}.xls` | Legacy Excel source | pandas/xlrd | One or more worksheets. |
| `{upload_id}_{safe_name}.xlsx` | Excel source | pandas/openpyxl | One or more worksheets. |
| `{upload_id}_{safe_name}.xlsb` | Binary Excel source | pandas/pyxlsb | One or more worksheets. |
| `{upload_id}_{safe_name}.xlsm` | Macro-enabled Excel source | pandas/openpyxl | Macros are treated as data input; the app does not execute them. |

## Key processes

The server validates extension, body size, and non-empty content before writing.
The pipeline reads the selected worksheet and never overwrites the original.

## Related documentation

- [User guide](../../docs/user-guide.md)
- [Data flow](../../docs/data-flow.md)
- [API reference](../../docs/api-reference.md)
