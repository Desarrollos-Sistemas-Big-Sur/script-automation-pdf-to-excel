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

# Términos de pago que indican pago de contado.
_CONTADO_TERMS: frozenset[str] = frozenset({
    "contado",
    "contado inmediato",
    "contado sin intereses",
    "consignacion",
    "consignación",
})

_ISBN_RE = re.compile(r'\b(97[89]\d{10})\b')

# Si Azure capturó menos del 50% de ISBNs presentes en raw_content -> activar fallback
_ISBN_COVERAGE_THRESHOLD = 0.5
# Si el PDF tiene más ISBNs que el raw_content en más de este factor -> Azure truncó el PDF
_PDF_TRUNCATION_FACTOR = 1.2


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
# Detección automática de calidad del resultado de Azure
# ---------------------------------------------------------------------------

def _isbns_in_raw(raw_content: str) -> set[str]:
    """ISBNs de 13 dígitos presentes en el raw_content devuelto por Azure."""
    return set(_ISBN_RE.findall(raw_content))


def _isbns_in_pdf(pdf_path: Path) -> set[str]:
    """ISBNs de 13 dígitos presentes en el PDF completo (todas las páginas).

    Usa pypdf para leer el texto directamente del archivo, independientemente
    de cuántas páginas haya procesado Azure.
    Devuelve conjunto vacío si pypdf no está disponible o falla.
    """
    if _PdfReader is None:
        return set()
    try:
        reader = _PdfReader(str(pdf_path))
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        return set(_ISBN_RE.findall(full_text))
    except Exception:
        return set()


def _isbns_in_lines(lines: list[DocumentLine]) -> set[str]:
    """ISBNs presentes en los ítems extraídos por Azure."""
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
    raw_content: str,
    azure_lines: list[DocumentLine],
    pdf_path: Path | None = None,
) -> bool:
    """Decide si conviene reemplazar los ítems de Azure con el parser de raw_content.

    Criterio 1 — cobertura baja en raw_content:
        Azure capturó menos del 50% de los ISBNs presentes en el raw_content.

    Criterio 2 — truncado detectado por pypdf:
        El PDF físico tiene más ISBNs únicos que el raw_content de Azure
        en un factor > 1.2, lo que indica que Azure no leyó todo el archivo.

    Si el raw_content no tiene ningún ISBN, no hay base para comparar
    y se respeta el resultado de Azure.
    """
    raw_isbns = _isbns_in_raw(raw_content)
    if not raw_isbns:
        return False

    # Criterio 1: cobertura de ISBNs en raw_content vs lo que extrajo Azure
    azure_isbns = _isbns_in_lines(azure_lines)
    coverage = len(azure_isbns) / len(raw_isbns)
    if coverage < _ISBN_COVERAGE_THRESHOLD:
        logger.warning(
            "Parser híbrido activado (cobertura baja): Azure capturó %d/%d ISBNs (%.0f%%).",
            len(azure_isbns), len(raw_isbns), coverage * 100,
        )
        return True

    # Criterio 2: el PDF físico tiene muchos más ISBNs que el raw_content -> Azure truncó
    if pdf_path is not None:
        pdf_isbns = _isbns_in_pdf(pdf_path)
        if pdf_isbns and len(pdf_isbns) > len(raw_isbns) * _PDF_TRUNCATION_FACTOR:
            logger.warning(
                "Parser híbrido activado (truncado Azure): PDF tiene %d ISBNs, "
                "raw_content solo %d (factor %.1fx).",
                len(pdf_isbns), len(raw_isbns),
                len(pdf_isbns) / len(raw_isbns),
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Parser de ítems directo desde raw_content
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> float | None:
    """Convierte string numérico a float.

    Maneja dos formatos:
      - Formato argentino/español: 1.234,56  (punto miles, coma decimal)
      - Formato anglosajón:        1,234.56  (coma miles, punto decimal)
    """
    if value is None:
        return None
    v = value.strip()
    # Si tiene tanto punto como coma, el último separador es el decimal
    if ',' in v and '.' in v:
        if v.rfind('.') > v.rfind(','):
            # Formato 1,234.56 -> quitar comas
            v = v.replace(',', '')
        else:
            # Formato 1.234,56 -> quitar puntos, reemplazar coma por punto
            v = v.replace('.', '').replace(',', '.')
    elif ',' in v:
        # Solo coma: puede ser decimal (45,00) o miles (1,234)
        # Si hay exactamente 2 dígitos tras la coma, es decimal
        parts = v.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            v = v.replace(',', '.')
        else:
            v = v.replace(',', '')
    try:
        return float(v)
    except (ValueError, AttributeError):
        return None


# Patrón Peyhache:
#   {ISBN-13}\n{Descripción}\n{cantidad}\nUnidad\n{precio}\n{descuento}\n{iva%}\n{total}
_PEYHACHE_ITEM_RE = re.compile(
    r'(97[89]\d{10})\n'
    r'([^\n]+)\n'
    r'([\d]+[,.]?\d*)\n'
    r'Unidad\n'
    r'([\d.,]+)\n'
    r'([\d.,]+)\n'
    r'[\d.,]+\n'
    r'([\d.,]+)',
)

# Patrón Guadal:
#   {código}\n{ISBN-13} {Descripción}\n{cantidad}\n{precio}\n-{descuento} %\n-{bonif}\n{total}
_GUADAL_ITEM_RE = re.compile(
    r'(\d{7})\n'
    r'(97[89]\d{10})\s+([^\n]+)\n'
    r'(\d+)\n'
    r'([\d.,]+)\n'
    r'-?([\d.,]+)\s*%\n'
    r'-?[\d.,]+\n'
    r'([\d.,]+)',
)


def _parse_items_from_raw_content(raw_content: str) -> list[DocumentLine]:
    """Parsea ítems directamente desde el texto crudo del PDF.

    Intenta primero patrón Peyhache, luego Guadal.
    Si ninguno produce resultados, devuelve lista vacía.
    """
    lines: list[DocumentLine] = []

    matches = _PEYHACHE_ITEM_RE.findall(raw_content)
    if matches:
        for index, (isbn, desc, qty, price, discount, total) in enumerate(matches, start=1):
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
        logger.info("Parser raw_content (Peyhache): %d ítems extraídos.", len(lines))
        return lines

    matches = _GUADAL_ITEM_RE.findall(raw_content)
    if matches:
        for index, (code, isbn, desc, qty, price, discount, total) in enumerate(matches, start=1):
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
        logger.info("Parser raw_content (Guadal): %d ítems extraídos.", len(lines))
        return lines

    logger.warning("Parser raw_content: ningún patrón conocido matchó. Se mantienen ítems de Azure.")
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
    """Agrega descuento a cada línea desde raw_content (in-place).
    No pisa line_discount ya seteado por el parser híbrido.
    """
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
    """Detecta copias (DUPLICADO/TRIPLICADO) y devuelve rango de páginas del original."""
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

        # Extraer ítems: Azure primero, parser híbrido si la cobertura es baja
        # o si Azure trunco el PDF (detectado comparando ISBNs del archivo vs raw_content)
        lines = _extract_lines(fields.get("Items"))

        if _should_use_raw_parser(raw_content, lines, pdf_path=file_path):
            raw_lines = _parse_items_from_raw_content(raw_content)
            if raw_lines:
                lines = raw_lines

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
