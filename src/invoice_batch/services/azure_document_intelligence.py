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
_ISBN_FULLMATCH = re.compile(r'^97[89]\d{10}$')

# Umbral de cobertura de ISBNs: si Azure capturó menos del 90% de los ISBNs
# presentes en el raw_content, se activa el parser híbrido completo.
# 90% es estricto — significa que Azure puede fallar en hasta 1 de cada 10
# ISBNs antes de que descartemos sus Items por completo.
_ISBN_COVERAGE_THRESHOLD = 0.9

# Ancla mínima para el enrichment por título: si el título tiene menos de
# 8 caracteres, no intentamos buscarlo en el raw_content (riesgo de falsos
# positivos demasiado alto).
_MIN_TITLE_ANCHOR_LEN = 8

# Cuántos caracteres del título usar como ancla de búsqueda.
_TITLE_ANCHOR_LEN = 20


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
    """Extrae los ítems del campo Items de Azure prebuilt-invoice.

    Mapea todos los campos estructurados disponibles incluyendo Discount,
    que Azure extrae nativamente.
    """
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
                    "line_discount": _amount(data.get("Discount")) or _safe_value(data.get("Discount")),
                },
            )
        )

    return items


# ---------------------------------------------------------------------------
# Detección automática de calidad del resultado de Azure
# ---------------------------------------------------------------------------

def _isbns_in_text(text: str) -> set[str]:
    """ISBNs de 13 dígitos presentes en un texto."""
    return set(_ISBN_RE.findall(text))


def _isbns_in_lines(lines: list[DocumentLine]) -> set[str]:
    """ISBNs presentes en los ítems extraídos por Azure."""
    found = set()
    for line in lines:
        code = line.values.get("product_code") or ""
        desc = line.values.get("description") or ""
        if _ISBN_FULLMATCH.match(str(code)):
            found.add(code)
        m = _ISBN_RE.search(desc)
        if m:
            found.add(m.group(1))
    return found


def _should_use_raw_parser(raw_content: str, azure_lines: list[DocumentLine]) -> bool:
    """Decide si usar el parser híbrido completo sobre el raw_content.

    Se activa cuando Azure capturó menos del 90% de los ISBNs presentes
    en su propio raw_content. Umbral alto porque preferimos el enrichment
    línea por línea (más quirúrgico) sobre reemplazar todos los ítems.
    """
    raw_isbns = _isbns_in_text(raw_content)
    if not raw_isbns:
        return False

    azure_isbns = _isbns_in_lines(azure_lines)
    coverage = len(azure_isbns) / len(raw_isbns)

    logger.info(
        "Cobertura ISBNs Azure: %d/%d (%.0f%%). Umbral: %.0f%%.",
        len(azure_isbns), len(raw_isbns), coverage * 100,
        _ISBN_COVERAGE_THRESHOLD * 100,
    )

    if coverage < _ISBN_COVERAGE_THRESHOLD:
        logger.warning(
            "Parser híbrido activado: Azure capturó solo %.0f%% de los ISBNs.",
            coverage * 100,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Parser de ítems directo desde raw_content (fallback completo)
# Se usa cuando Azure no extrajo bien los Items en general.
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> float | None:
    """Convierte string numérico a float.

    Maneja:
      - Formato argentino/español: 1.234,56  (punto miles, coma decimal)
      - Formato anglosajón:        1,234.56  (coma miles, punto decimal)
      - Con signo pesos:            $ 1.234,56
    """
    if value is None:
        return None
    v = value.strip().lstrip('$').strip()
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


# Patrón Peyhache — formato Azure raw_content (S0, multi-línea):
#   {ISBN-13}\n{descripción}\n{cantidad}\nUnidad\n{precio}\n{descuento}\n{iva%}\n{total}
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


def _parse_items_from_raw_content(raw_content: str) -> list[DocumentLine]:
    """Parsea ítems directamente desde el raw_content de Azure.

    Solo se usa como fallback cuando Azure no extrajo bien los Items.
    Intenta primero patrón Peyhache, luego Guadal.
    Devuelve lista vacía si ningún patrón matchea.
    """
    # Peyhache
    matches = _PEYHACHE_ITEM_RE.findall(raw_content)
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
        logger.info("Parser raw_content (Peyhache): %d ítems extraídos.", len(lines))
        return lines

    # Guadal
    matches = _GUADAL_ITEM_RE.findall(raw_content)
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
        logger.info("Parser raw_content (Guadal): %d ítems extraídos.", len(lines))
        return lines

    logger.warning("Parser raw_content: ningún patrón matcheo. Se mantienen ítems de Azure.")
    return []


# ---------------------------------------------------------------------------
# Enrichment genérico línea por línea
# ---------------------------------------------------------------------------

# Número de 9 a 13 dígitos: cubre ISBN-13, ISBN-10 y códigos de proveedor
# numéricos. En facturas de libros argentinas la primera o segunda columna
# siempre es el código o ISBN.
_CODE_RE = re.compile(r'\b(\d{9,13})\b')

# Descuento: número de 1-3 dígitos, opcionalmente con decimales o signo %.
# Cubre: 47.00 | 50 | -50 % | 47,00
_DISCOUNT_RE = re.compile(r'-?(\d{1,3}(?:[.,]\d{1,2})?)\s*%?')


def _enrich_line_from_raw_content(
    raw_content: str,
    description: str,
) -> tuple[str | None, float | None]:
    """Busca código/ISBN y descuento en el raw_content usando el título como ancla.

    Estrategia genérica — no asume nada del proveedor:
      1. Tomar los primeros _TITLE_ANCHOR_LEN chars del título como ancla.
      2. Buscar esa cadena en el raw_content.
      3. Mirar hacia atrás desde el título: el número más largo de 9-13
         dígitos inmediatamente antes es el código/ISBN.
      4. Mirar hacia adelante desde el título: el primer número que parezca
         un porcentaje de descuento (1-3 dígitos, opcionalmente decimales).

    Devuelve (codigo, descuento). Cualquiera puede ser None si no se encontró.
    """
    if not description or len(description) < _MIN_TITLE_ANCHOR_LEN:
        return None, None

    anchor = re.escape(description[:_TITLE_ANCHOR_LEN])
    m = re.search(anchor, raw_content)
    if not m:
        return None, None

    pos = m.start()

    # --- Código/ISBN: buscar hacia atrás en las 2 líneas previas ---
    # Tomamos el texto desde 200 chars antes hasta el inicio del título
    before = raw_content[max(0, pos - 200):pos]
    # Quedarnos solo con las últimas 2 líneas antes del título
    before_lines = before.rstrip('\n').split('\n')
    before_window = '\n'.join(before_lines[-2:]) if before_lines else ''

    code_found = None
    # Buscar el número más largo de 9-13 dígitos en esas líneas
    code_matches = _CODE_RE.findall(before_window)
    if code_matches:
        # Preferir el más largo (más probable que sea ISBN o código de proveedor)
        code_found = max(code_matches, key=len)

    # --- Descuento: buscar hacia adelante en las 4 líneas siguientes ---
    after = raw_content[pos + len(m.group()):pos + len(m.group()) + 300]
    after_lines = after.lstrip('\n').split('\n')
    after_window = '\n'.join(after_lines[:4]) if after_lines else ''

    discount_found = None
    for line in after_lines[:4]:
        line = line.strip()
        # Buscar líneas que sean solo un número de descuento (ej: "47.00" o "-50 %")
        dm = re.fullmatch(r'-?(\d{1,3}(?:[.,]\d{1,2})?)\s*%?', line)
        if dm:
            try:
                val = float(dm.group(1).replace(',', '.'))
                # Rango razonable de descuento: 1% a 99%
                if 1 <= val <= 99:
                    discount_found = val
                    break
            except ValueError:
                continue

    return code_found, discount_found


def _find_isbn_in_description(description: str) -> str | None:
    """Extrae el ISBN de la descripción si Azure lo puso ahí en lugar de en ProductCode."""
    if not description:
        return None
    m = _ISBN_RE.search(description)
    return m.group(1) if m else None


def _enrich_lines_with_discounts(lines: list[DocumentLine], raw_content: str) -> None:
    """Colchón genérico: completa código/ISBN y descuento faltantes (in-place).

    Para cada línea donde product_code o line_discount son None:
      1. Intentar extraer el ISBN de la descripción (Azure a veces los mezcla).
      2. Si sigue faltando código o descuento, usar el título como ancla
         en el raw_content y extraer los valores del bloque circundante.
         Esta lógica es genérica — no asume nada del proveedor.
    """
    enriched_code = 0
    enriched_discount = 0

    for line in lines:
        product_code = line.values.get("product_code") or ""
        description = line.values.get("description") or ""
        needs_code = not product_code or not re.match(r'^\d{9,13}$', str(product_code))
        needs_discount = line.values.get("line_discount") is None

        # Paso 1: ISBN oculto en la descripción
        if needs_code:
            isbn_from_desc = _find_isbn_in_description(description)
            if isbn_from_desc:
                line.values["product_code"] = isbn_from_desc
                product_code = isbn_from_desc
                needs_code = False

        # Paso 2: enrichment genérico por título como ancla
        if needs_code or needs_discount:
            raw_code, raw_discount = _enrich_line_from_raw_content(raw_content, description)

            if needs_code and raw_code:
                line.values["product_code"] = raw_code
                enriched_code += 1

            if needs_discount and raw_discount is not None:
                line.values["line_discount"] = raw_discount
                enriched_discount += 1

    if enriched_code or enriched_discount:
        logger.info(
            "Enrichment genérico: %d códigos y %d descuentos rescatados desde raw_content.",
            enriched_code, enriched_discount,
        )


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
        r'Fecha\s+(?:de\s+)?Vto\.?\s*(?:\s*de\s+CAE)?\s*\.?:?\s*[\s\n]*(\d{2}[/-]\d{2}[/-]\d{4})',
        content,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _parse_document_letter(content: str) -> str | None:
    match = re.search(r'(?:^|\n)([ABC])\n', content)
    return match.group(1) if match else None


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

        # Flujo: Azure extrae Items → si cobertura < 90% se descarta y
        # se usa el parser híbrido completo → siempre se aplica el
        # enrichment genérico línea por línea para campos faltantes.
        lines = _extract_lines(fields.get("Items"))

        if _should_use_raw_parser(raw_content, lines):
            parsed_lines = _parse_items_from_raw_content(raw_content)
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
