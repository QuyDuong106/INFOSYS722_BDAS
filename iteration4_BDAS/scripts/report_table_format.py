"""Bold best displayed metric means consistently in report result tables."""
import re
from decimal import Decimal

def bold_best_cells(rows, headers):
    result = [[re.sub(r'\\textbf\{([^{}]*)\}', r'\1', str(cell)) for cell in row] for row in rows]
    for col, header in enumerate(headers[1:], 1):
        if header in {'Fields', 'Features', 'Seed', 'Rows'}:
            continue
        values = []
        for index, row in enumerate(result):
            match = re.match(r'^([+-]?\d+(?:\.\d+)?)', row[col])
            if match:
                value = Decimal(match.group(1))
                values.append((index, value, match))
        if not values:
            continue
        lower = any(word in header.lower() for word in ['error', 'loss', 'false-negative', 'fnr'])
        best = (min if lower else max)(v for _, v, _ in values)
        for index, value, match in values:
            if value == best:
                cell = result[index][col]
                result[index][col] = r'\textbf{' + match.group(1) + '}' + cell[match.end():]
    return result
