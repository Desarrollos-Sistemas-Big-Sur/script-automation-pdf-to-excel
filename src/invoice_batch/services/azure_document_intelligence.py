from __future__ import annotations

import logging
import re
from pathlib import Path

from invoice_batch.config import AzureConfig
from invoice_batch.domain.models import DocumentLine, ExtractedDocument

logger = logging.getLogger("invoice_batch.azure")

try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:  # pragma: no cover
    DocumentAnalysisClient = None
    AzureKeyCredential = None

try:
    from pypdf import PdfReader as _PdfReader
except ImportError:  # pragma: no cover
    _PdfReader = None

_CONTADO_TERMS: frozenset[str] = frozenset({
    "contado",
    "contado inmediato",
    "contado sin intereses",
    "consignacion",
    "consignación",
})

_ISBN_RE = re.compile(r'\b(97[89]\d{10})\b')
_ISBN_COVERAGE_THRESHOLD = 0.5


def _safe_value(field):
    return field.value if field else None


def _safe_text(field):
    value = _safe_value(field)
    return str(value) if value is not None else None


def _first_present_text(fields, *names: str):
    for name in names:
        value = _safe_text(fields.get(name))
        if value not in (None, ""):
            return value
    return None


def _amount(field):
    if not field or not field.value:
        return None
    return getattr(field.value, "amount", None)


def _currency(field):
    if not field or not field.value:
        return None
    return getattr(field.value, "currency_code", None) or getattr(field.value, "symbol", None)


def _content(field):
    return getattr(field, "content", None) if field else None


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split())


def _format_address(field):
    value = _safe_value(field)
    if value is None:
        return None
    if isinstance(value, str):
        return value

    parts = [
        getattr(value, "street_address", None),
        getattr(value, "road", None),
        getattr(value, "house_number", None),
        getattr(value, "unit", None),
        getattr(value, "city", None),
        getattr(value, "state", None),
        getattr(value, "postal_code", None),
        getattr(value, "country_region", None),
    ]

    clean_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        normalized = str(part).strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        clean_parts.append(normalized)

    if clean_parts:
        return ", ".join(clean_parts)

    return _content(field)


def _extract_lines(items_field) -> list[DocumentLine]:
    items: list[DocumentLine] = []
    if not items_field or not items_field.value:
        return items

    for index, item in enumerate(items_field.value, start=1):
        data = item.value
        items.append(
            DocumentLine(
                line_number=index,
                values={
                    "description": _safe_text(data.get("Description")) or "S/D",
                    "quantity": _safe_value(data.get("Quantity")),
                    "unit_price": _amount(data.get("UnitPrice")),
                    "line_total": _amount(data.get("Amount")),
                    "product_code": _safe_text(data.get("ProductCode")),
                    "item_date": _safe_text(data.get("Date")),
                },
            )
        )

    return items


# ---------------------------------------------------------------------------
# Extracción de texto completo del PDF via pypdf
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> str:
    """Lee el texto completo del PDF usando pypdf (todas las páginas)."""
    if _PdfReader is None:
        logger.warning("pypdf no disponible: %s", pdf_path.name)
        return ""
    try:
        reader = _PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        logger.info(
            "pypdf leyó %d páginas de '%s' (%d chars).",
            len(reader.pages), pdf_path.name, len(text),
        )
        return text
    except Exception as exc:
        logger.warning("pypdf falló leyendo '%s': %s", pdf_path.name, exc)
        return ""


def _isbns_in_text(text: str) -> set[str]:
    return set(_ISBN_RE.findall(text))


def _isbns_in_lines(lines: list[DocumentLine]) -> set[str]:
    found = set()
    for line in lines:
        code = line.values.get("product_code") or ""
        desc = line.values.get("description") or ""
        if re.fullmatch(r'97[89]\d{10}', code):
            found.add(code)
        m = _ISBN_RE.search(desc)
        if m:
            found.add(m.group(1))
    return found


def _should_use_raw_parser(
    azure_raw_content: str,
    pdf_full_text: str,
    azure_lines: list[DocumentLine],
) -> bool:
    """Decide si usar el parser de texto crudo en lugar de los ítems de Azure.

    Criterio 1 — truncado por pypdf:
        El texto del PDF completo tiene más ISBNs únicos que el raw_content
        de Azure -> Azure no leyó todo el archivo.

    Criterio 2 — cobertura baja en raw_content:
        Azure capturó menos del 50% de los ISBNs de su propio raw_content.

    Nota especial: si el PDF tiene ISBNs pero el raw_content no (PDF con
    texto no seleccionable que Azure lee vía OCR), igual se activa el
    parser para intentar extraer desde el texto del PDF.
    """
    raw_isbns = _isbns_in_text(azure_raw_content)
    pdf_isbns = _isbns_in_text(pdf_full_text)

    logger.info(
        "ISBNs únicos — PDF completo: %d | raw_content Azure: %d",
        len(pdf_isbns), len(raw_isbns),
    )

    # Sin ISBNs en ninguna fuente: no hay base para comparar
    if not raw_isbns and not pdf_isbns:
        return False

    # Criterio 1: pypdf encontró más ISBNs que Azure en su raw_content
    if pdf_isbns and len(pdf_isbns) > len(raw_isbns):
        logger.warning(
            "Parser híbrido activado (truncado Azure): PDF=%d ISBNs, Azure raw=%d ISBNs.",
            len(pdf_isbns), len(raw_isbns),
        )
        return True

    # Criterio 1b: pypdf no encontró ISBNs pero Azure sí los tiene en raw_content
    # y la cobertura en Items es baja -> el PDF tiene layout especial, confiar en raw_content
    if raw_isbns and not pdf_isbns:
        azure_isbns = _isbns_in_lines(azure_lines)
        coverage = len(azure_isbns) / len(raw_isbns)
        if coverage < _ISBN_COVERAGE_THRESHOLD:
            logger.warning(
                "Parser híbrido activado (cobertura baja, PDF con layout especial): "
                "Azure capturó %d/%d ISBNs (%.0f%%).",
                len(azure_isbns), len(raw_isbns), coverage * 100,
            )
            return True
        return False

    # Criterio 2: cobertura baja en raw_content
    if raw_isbns:
        azure_isbns = _isbns_in_lines(azure_lines)
        coverage = len(azure_isbns) / len(raw_isbns)
        if coverage < _ISBN_COVERAGE_THRESHOLD:
            logger.warning(
                "Parser híbrido activado (cobertura baja): Azure capturó %d/%d ISBNs (%.0f%%).",
                len(azure_isbns), len(raw_isbns), coverage * 100,
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Parser de ítems directo desde texto crudo
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> float | None:
    """Convierte string numérico a float.

    Maneja:
      - Formato argentino/español: 1.234,56  (punto miles, coma decimal)
      - Formato anglosajón:        1,234.56  (coma miles, punto decimal)
    """
    if value is None:
        return None
    v = value.strip()
    if ',' in v and '.' in v:
        if v.rfind('.') > v.rfind(','):
            v = v.replace(',', '')
        else:
            v = v.replace('.', '').replace(',', '.')
    elif ',' in v:
        parts = v.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            v = v.replace(',', '.')
        else:
            v = v.replace(',', '')
    try:
        return float(v)
    except (ValueError, AttributeError):
        return None


# Patrón Peyhache — formato pypdf (una línea por ítem):
#   {descuento},{ISBN-13} {descripción} {cantidad} {total}{precio}Unidad {iva}
#
# Ejemplo real:
#   45,009789878281308 Kintsugi 1,00 13.530,0024.600,00Unidad 0,00
#
# Grupos capturados: isbn, descripción, cantidad, total, precio, descuento
_PEYHACHE_PDF_ITEM_RE = re.compile(
    r'(\d{1,3},\d{2})'            # descuento (ej: 45,00 o 59,00)
    r'(97[89]\d{10})\s+'          # ISBN-13
    r'(.+?)\s+'                   # descripción (non-greedy)
    r'(\d+,\d{2})\s+'             # cantidad (ej: 1,00 o 4,00)
    r'([\d.]+,\d{2})'             # total línea (ej: 13.530,00)
    r'([\d.]+,\d{2})'             # precio unitario (ej: 24.600,00) -- pegado al total
    r'Unidad\s+'
    r'[\d,]+',                    # IVA % (ignorado)
)

# Patrón Peyhache — formato Azure raw_content (multi-línea):
#   {ISBN-13}\n{descripción}\n{cantidad}\nUnidad\n{precio}\n{descuento}\n{iva}\n{total}
_PEYHACHE_AZURE_ITEM_RE = re.compile(
    r'(97[89]\d{10})\n'
    r'([^\n]+)\n'
    r'([\d]+[,.]?\d*)\n'
    r'Unidad\n'
    r'([\d.,]+)\n'
    r'([\d.,]+)\n'
    r'[\d.,]+\n'
    r'([\d.,]+)',
)

# Patrón Guadal — formato Azure raw_content:
#   {código}\n{ISBN-13} {descripción}\n{cantidad}\n{precio}\n-{descuento} %\n-{bonif}\n{total}
_GUADAL_ITEM_RE = re.compile(
    r'(\d{7})\n'
    r'(97[89]\d{10})\s+([^\n]+)\n'
    r'(\d+)\n'
    r'([\d.,]+)\n'
    r'-?([\d.,]+)\s*%\n'
    r'-?[\d.,]+\n'
    r'([\d.,]+)',
)


def _parse_items_peyhache_pdf(text: str) -> list[DocumentLine]:
    """Parsea ítems de Peyhache desde texto pypdf (una línea por ítem)."""
    lines: list[DocumentLine] = []
    for index, m in enumerate(_PEYHACHE_PDF_ITEM_RE.finditer(text), start=1):
        discount, isbn, desc, qty, total, price = m.groups()
        lines.append(DocumentLine(
            line_number=index,
            values={
                "description": desc.strip(),
                "quantity": _parse_number(qty),
                "unit_price": _parse_number(price),
                "line_total": _parse_number(total),
                "product_code": isbn,
                "item_date": None,
                "line_discount": _parse_number(discount),
            },
        ))
    return lines


def _parse_items_from_text(text: str, source: str = "") -> list[DocumentLine]:
    """Parsea ítems desde texto crudo.

    Intenta en orden: Peyhache PDF, Peyhache Azure, Guadal.
    Devuelve lista vacía si ningún patrón matchea.
    """
    # Peyhache formato pypdf
    lines = _parse_items_peyhache_pdf(text)
    if lines:
        logger.info("Parser %s (Peyhache PDF): %d ítems.", source, len(lines))
        return lines

    # Peyhache formato Azure raw_content
    matches = _PEYHACHE_AZURE_ITEM_RE.findall(text)
    if matches:
        lines = [
            DocumentLine(
                line_number=i,
                values={
                    "description": desc.strip(),
                    "quantity": _parse_number(qty),
                    "unit_price": _parse_number(price),
                    "line_total": _parse_number(total),
                    "product_code": isbn,
                    "item_date": None,
                    "line_discount": _parse_number(discount),
                },
            )
            for i, (isbn, desc, qty, price, discount, total)
            in enumerate(matches, start=1)
        ]
        logger.info("Parser %s (Peyhache Azure): %d ítems.", source, len(lines))
        return lines

    # Guadal
    matches = _GUADAL_ITEM_RE.findall(text)
    if matches:
        lines = [
            DocumentLine(
                line_number=i,
                values={
                    "description": desc.strip(),
                    "quantity": _parse_number(qty),
                    "unit_price": _parse_number(price),
                    "line_total": _parse_number(total),
                    "product_code": isbn,
                    "item_date": None,
                    "line_discount": _parse_number(discount),
                },
            )
            for i, (code, isbn, desc, qty, price, discount, total)
            in enumerate(matches, start=1)
        ]
        logger.info("Parser %s (Guadal): %d ítems.", source, len(lines))
        return lines

    logger.warning("Parser %s: ningún patrón matchó.", source)
    return []


# ---------------------------------------------------------------------------
# Parsing complementario sobre texto crudo
# ---------------------------------------------------------------------------

def _parse_cae(content: str) -> str | None:
    match = re.search(
        r'C\.?A\.?E\.?\s*(?:N[\u00b0\u00ba])?\s*:?\s*[\s\n]*(\d{10,})',
        content,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _parse_cae_due_date(content: str) -> str | None:
    match = re.search(
        r'Fecha\s+(?:de\s+)?Vto\.?\s*(?:\s*de\s+CAE)?\s*\.?:?\s*[\s\n]*(\d{2}/\d{2}/\d{4})',
        content,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _parse_document_letter(content: str) -> str | None:
    match = re.search(r'(?:^|\n)([ABC])\n', content)
    return match.group(1) if match else None


def _parse_pct(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _find_discount_for_item(raw_content: str, product_code: str) -> float | None:
    escaped = re.escape(product_code)

    m = re.search(
        escaped + r"\n[^\n]+\n[^\n]+\nUnidad\n[^\n]+\n([\d,]+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    m = re.search(
        escaped + r"\n\d+\n[^\n]+\n[^\n]+\n(\d+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    m = re.search(
        escaped + r"\n[^\n]+\n[^\n]+ unidades\n[^\n]+ ([\d,]+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    return None


def _enrich_lines_with_discounts(lines: list[DocumentLine], raw_content: str) -> None:
    for line in lines:
        if line.values.get("line_discount") is not None:
            continue
        product_code = line.values.get("product_code")
        line.values["line_discount"] = (
            _find_discount_for_item(raw_content, product_code)
            if product_code
            else None
        )


# ---------------------------------------------------------------------------
# Parser de acuses de devolución
# ---------------------------------------------------------------------------

_DEVOLUCION_ITEM_RE = re.compile(
    r'(\d{13})\n([^\n]+)\n(\d+)\n\d+\n\d+\n\d+',
)


def _parse_devolucion_items(raw_content: str) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for isbn, titulo, recibida in _DEVOLUCION_ITEM_RE.findall(raw_content):
        result[isbn] = (titulo, int(recibida))
    return result


def _enrich_lines_from_devolucion(lines: list[DocumentLine], raw_content: str) -> None:
    parsed = _parse_devolucion_items(raw_content)
    if not parsed:
        return

    title_lookup: dict[str, str] = {
        titulo.upper().strip(): isbn
        for isbn, (titulo, _) in parsed.items()
    }

    for line in lines:
        if line.values.get("product_code"):
            continue

        desc = (line.values.get("description") or "").upper().strip()
        isbn = title_lookup.get(desc)
        if isbn:
            _, recibida = parsed[isbn]
            line.values["product_code"] = isbn
            if not line.values.get("quantity"):
                line.values["quantity"] = recibida


def _parse_document_subtype(content: str) -> str | None:
    content_lower = content.lower()
    if "nota de cr\u00e9dito" in content_lower or "nota de credito" in content_lower:
        return "Nota de Crédito"
    if "nota de d\u00e9bito" in content_lower or "nota de debito" in content_lower:
        return "Nota de Débito"
    if (
        "nota de devoluci\u00f3n" in content_lower
        or "nota de devolucion" in content_lower
        or "diferencias en las devoluciones" in content_lower
        or "diferencias en devoluciones" in content_lower
        or "devolucion de consignacion" in content_lower
        or "devoluci\u00f3n de consignaci\u00f3n" in content_lower
        or "remito de devolucion" in content_lower
        or "remito de devoluci\u00f3n" in content_lower
    ):
        return "Acuse de Devolución"
    if re.search(r'(?<!no v[a\u00e1]lido como )\bfactura\b', content_lower):
        return "Factura"
    return None


_COPY_MARKERS = re.compile(r'\b(DUPLICADO|TRIPLICADO|CUADRUPLICADO)\b', re.IGNORECASE)


def _original_pages_param(pdf_path: Path) -> str | None:
    if _PdfReader is None:
        return None
    try:
        reader = _PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if _COPY_MARKERS.search(text):
                last_original = i - 1
                if last_original < 1:
                    logger.warning(
                        "Marca de copia en página 1 de '%s'. Se omite truncado.",
                        pdf_path.name,
                    )
                    return None
                logger.info(
                    "Marca de copia en página %d de '%s'. Azure procesará páginas 1-%d.",
                    i, pdf_path.name, last_original,
                )
                return str(last_original) if last_original == 1 else f"1-{last_original}"
    except Exception:
        pass
    return None


class AzureDocumentIntelligenceExtractor:
    def __init__(self, config: AzureConfig) -> None:
        self.config = config
        self._client = None

    def _get_client(self):
        if not self.config.endpoint or not self.config.key:
            raise ValueError(
                "Faltan credenciales Azure. Configurar AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "y AZURE_DOCUMENT_INTELLIGENCE_KEY."
            )
        if DocumentAnalysisClient is None or AzureKeyCredential is None:
            raise RuntimeError("Dependencias de Azure no disponibles en el entorno.")
        if self._client is None:
            self._client = DocumentAnalysisClient(
                endpoint=self.config.endpoint,
                credential=AzureKeyCredential(self.config.key),
            )
        return self._client

    def extract(self, file_path: Path, document_type: str) -> tuple[ExtractedDocument, dict]:
        client = self._get_client()

        # Leer el PDF completo con pypdf ANTES de Azure.
        # Esto nos da el texto de todas las páginas y sirve como fuente
        # para detectar truncado y como base para el parser híbrido.
        pdf_full_text = _extract_pdf_text(file_path)

        pages = _original_pages_param(file_path) or self.config.pages
        kwargs = {}
        if pages:
            kwargs["pages"] = pages

        with file_path.open("rb") as handle:
            poller = client.begin_analyze_document(self.config.model_id, document=handle, **kwargs)
            result = poller.result()

        if not result.documents:
            raise ValueError("Azure no devolvio documentos analizados para ese archivo.")

        az_doc = result.documents[0]
        fields = az_doc.fields
        raw_content: str = result.content or ""

        payment_terms = _first_present_text(fields, "PaymentTerms", "PaymentTerm")

        raw_due_date = _safe_text(fields.get("DueDate"))
        if payment_terms and payment_terms.strip().lower() in _CONTADO_TERMS:
            invoice_due_date = None
        else:
            invoice_due_date = raw_due_date

        fields_payload = {
            "invoice_id": _safe_value(fields.get("InvoiceId")),
            "purchase_order": _safe_value(fields.get("PurchaseOrder")),
            "issue_date": _safe_text(fields.get("InvoiceDate")),
            "invoice_due_date": invoice_due_date,
            "payment_terms": payment_terms,
            "cae": _parse_cae(raw_content),
            "cae_due_date": _parse_cae_due_date(raw_content),
            "document_letter": _parse_document_letter(raw_content),
            "document_subtype": _parse_document_subtype(raw_content),
            "vendor_name": _clean_text(_safe_text(fields.get("VendorName"))),
            "vendor_address": _format_address(fields.get("VendorAddress")),
            "vendor_tax_id": _safe_value(fields.get("VendorTaxId")),
            "customer_name": _safe_value(fields.get("CustomerName")),
            "customer_id": _safe_value(fields.get("CustomerId")),
            "customer_address": _format_address(fields.get("CustomerAddress")),
            "subtotal_amount": _amount(fields.get("SubTotal")),
            "tax_amount": _amount(fields.get("TotalTax")),
            "total_amount": _amount(fields.get("InvoiceTotal")),
            "currency": _currency(fields.get("InvoiceTotal")),
            "raw_total_content": _content(fields.get("InvoiceTotal")),
        }

        lines = _extract_lines(fields.get("Items"))

        if _should_use_raw_parser(raw_content, pdf_full_text, lines):
            # Primero intentar con el texto completo del PDF (cubre truncado)
            parsed_lines = _parse_items_from_text(pdf_full_text, source="PDF") if pdf_full_text else []
            # Si el PDF no dio resultados, intentar con el raw_content de Azure
            if not parsed_lines:
                parsed_lines = _parse_items_from_text(raw_content, source="Azure")
            if parsed_lines:
                lines = parsed_lines

        _enrich_lines_with_discounts(lines, raw_content)
        _enrich_lines_from_devolucion(lines, raw_content)

        document = ExtractedDocument(
            source_file=file_path.name,
            document_type=document_type,
            fields=fields_payload,
            lines=lines,
        )

        raw_payload = {
            "source_file": document.source_file,
            "document_type": document.document_type,
            "fields": document.fields,
            "line_count": len(document.lines),
            "raw_content": raw_content,
        }
        return document, raw_payload
