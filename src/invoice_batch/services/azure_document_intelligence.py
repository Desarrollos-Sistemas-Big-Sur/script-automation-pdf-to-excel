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
# En estos casos invoice_due_date no existe como concepto y debe ser null.
_CONTADO_TERMS: frozenset[str] = frozenset({
    "contado",
    "contado inmediato",
    "contado sin intereses",
    "consignacion",
    "consignación",
})

# Regex para detectar ISBNs de 13 dígitos en el raw_content
_ISBN_RE = re.compile(r'\b(97[89]\d{10})\b')

# Umbral: si Azure extrajo menos del 50% de los ISBNs presentes en el
# raw_content, se activa el parser de fallback desde raw_content.
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
    """Normaliza espacios y saltos de línea en strings extraídos por Azure."""
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
    """Devuelve el conjunto de ISBNs de 13 dígitos presentes en el raw_content."""
    return set(_ISBN_RE.findall(raw_content))


def _isbns_in_lines(lines: list[DocumentLine]) -> set[str]:
    """Devuelve el conjunto de ISBNs presentes en los ítems extraídos por Azure."""
    found = set()
    for line in lines:
        code = line.values.get("product_code") or ""
        desc = line.values.get("description") or ""
        # El ISBN puede estar en product_code directamente
        if re.fullmatch(r'97[89]\d{10}', code):
            found.add(code)
        # O puede haber quedado pegado al inicio de la descripción
        m = _ISBN_RE.search(desc)
        if m:
            found.add(m.group(1))
    return found


def _should_use_raw_parser(raw_content: str, azure_lines: list[DocumentLine]) -> bool:
    """Decide si conviene reemplazar los ítems de Azure con el parser de raw_content.

    Criterios (basta con que se cumpla uno):
    - Hay ISBNs en el raw_content pero Azure no extrajo ninguno en los ítems.
    - Azure capturó menos del 50% de los ISBNs presentes en el raw_content.

    Si el raw_content no tiene ningún ISBN (factura sin libros, por ejemplo),
    se respeta el resultado de Azure sin tocar nada.
    """
    raw_isbns = _isbns_in_raw(raw_content)
    if not raw_isbns:
        return False  # No hay ISBNs que comparar, Azure es la única fuente

    azure_isbns = _isbns_in_lines(azure_lines)
    coverage = len(azure_isbns) / len(raw_isbns)

    if coverage < _ISBN_COVERAGE_THRESHOLD:
        logger.warning(
            "Parser híbrido activado: Azure capturó %d/%d ISBNs (%.0f%%). "
            "Se usará el parser de raw_content para los ítems.",
            len(azure_isbns), len(raw_isbns), coverage * 100,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Parser de ítems directo desde raw_content
# Cubre los formatos observados en Peyhache y Guadal.
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> float | None:
    """Convierte string numérico con punto o coma decimal a float."""
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


# Patrón Peyhache:
#   {ISBN-13}\n{Descripción}\n{cantidad},00\nUnidad\n{precio},00\n{descuento},00\n{iva%}\n{total},00
_PEYHACHE_ITEM_RE = re.compile(
    r'(97[89]\d{10})\n'           # ISBN
    r'([^\n]+)\n'                  # Descripción
    r'([\d]+[,.]?\d*)\n'           # Cantidad
    r'Unidad\n'                    # Unidad de medida (ancla)
    r'([\d.,]+)\n'                 # Precio unitario
    r'([\d.,]+)\n'                 # % Descuento
    r'[\d.,]+\n'                   # IVA% (ignorado, siempre 0)
    r'([\d.,]+)',                  # Total línea
)

# Patrón Guadal:
#   {código corto}\n{ISBN-13} {Descripción}\n{cantidad}\n{precio}\n-{descuento} %\n-{bonif}\n{total}
_GUADAL_ITEM_RE = re.compile(
    r'(\d{7})\n'                               # Código corto proveedor
    r'(97[89]\d{10})\s+([^\n]+)\n'             # ISBN + Descripción en la misma línea
    r'(\d+)\n'                                 # Cantidad
    r'([\d.,]+)\n'                             # Precio unitario
    r'-?([\d.,]+)\s*%\n'                       # % Descuento (con o sin signo negativo)
    r'-?[\d.,]+\n'                             # Bonificación (ignorada)
    r'([\d.,]+)',                              # Total línea
)


def _parse_items_from_raw_content(raw_content: str) -> list[DocumentLine]:
    """Parsea ítems directamente desde el texto crudo del PDF.

    Intenta primero el patrón Peyhache, luego el patrón Guadal.
    Si ninguno produce resultados, devuelve lista vacía (Azure mantiene el control).
    """
    lines: list[DocumentLine] = []

    # Intentar patrón Peyhache
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

    # Intentar patrón Guadal
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
# Parsing complementario sobre texto crudo (result.content)
# Azure prebuilt-invoice no devuelve estos campos como fields estructurados.
# ---------------------------------------------------------------------------

def _parse_cae(content: str) -> str | None:
    """Extrae el número de CAE desde el texto crudo.

    Cubre formatos observados:
      - 'CAE N°: 86139139214749'       (TRAPEZOIDE)
      - 'C.A.E. : 86117359422834'      (Ediciones Continente)
      - 'CAE Nº:\\n86107042866417'     (PEYHACHE)
    """
    match = re.search(
        r'C\.?A\.?E\.?\s*(?:N[\u00b0\u00ba])?\s*:?\s*[\s\n]*(\d{10,})',
        content,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _parse_cae_due_date(content: str) -> str | None:
    """Extrae la fecha de vencimiento del CAE desde el texto crudo.

    Cubre formatos observados:
      - 'Fecha de Vto. de CAE: 05/04/2026'   (TRAPEZOIDE)
      - 'Fecha Vto .: 22/03/2026'             (Ediciones Continente)
      - 'Fecha de Vto. de CAE:\\n20/03/2026'  (PEYHACHE)
    """
    match = re.search(
        r'Fecha\s+(?:de\s+)?Vto\.?\s*(?:\s*de\s+CAE)?\s*\.?:?\s*[\s\n]*(\d{2}/\d{2}/\d{4})',
        content,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _parse_document_letter(content: str) -> str | None:
    """Extrae la letra del comprobante (A, B o C) desde el texto crudo.

    En los PDFs analizados la letra aparece como una línea aislada
    de un solo carácter antes o después del tipo de comprobante.
    """
    match = re.search(r'(?:^|\n)([ABC])\n', content)
    return match.group(1) if match else None


def _parse_pct(value: str) -> float | None:
    """Convierte string de porcentaje ('45,00' o '45') a float."""
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _find_discount_for_item(raw_content: str, product_code: str) -> float | None:
    """
    Busca el descuento de un ítem en el texto crudo usando el código de producto como ancla.

    Cubre tres layouts observados:

    PEYHACHE (ISBN-13):
        {code}\\n{desc}\\n{qty}\\nUnidad\\n{precio}\\n{descuento}\\n{iva%}\\n{total}

    Ediciones Continente (cód. alfanumérico tipo OLA079):
        {code}\\n{qty}\\n{desc}\\n{precio}\\n{descuento}\\n{neto}

    TRAPEZOIDE (cód. numérico corto tipo 010):
        {code}\\n{desc}\\n{qty} unidades\\n{precio} {descuento}\\n
    """
    escaped = re.escape(product_code)

    # PEYHACHE: precio y descuento en líneas separadas, unidad = "Unidad"
    m = re.search(
        escaped + r"\n[^\n]+\n[^\n]+\nUnidad\n[^\n]+\n([\d,]+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    # Ediciones Continente: qty en segunda línea, descuento entero tras el precio
    m = re.search(
        escaped + r"\n\d+\n[^\n]+\n[^\n]+\n(\d+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    # TRAPEZOIDE: precio y descuento en la misma línea separados por espacio
    m = re.search(
        escaped + r"\n[^\n]+\n[^\n]+ unidades\n[^\n]+ ([\d,]+)\n",
        raw_content,
    )
    if m:
        return _parse_pct(m.group(1))

    return None


def _enrich_lines_with_discounts(lines: list[DocumentLine], raw_content: str) -> None:
    """Agrega el descuento a cada línea desde el texto crudo (in-place).

    Solo se aplica a líneas que no tienen line_discount ya seteado
    (el parser de raw_content puede haberlo extraído directamente).
    """
    for line in lines:
        if line.values.get("line_discount") is not None:
            continue  # Ya extraído por el parser híbrido, no pisar
        product_code = line.values.get("product_code")
        line.values["line_discount"] = (
            _find_discount_for_item(raw_content, product_code)
            if product_code
            else None
        )


# ---------------------------------------------------------------------------
# Parser de acuses de devolución (formato multi-columna)
# ---------------------------------------------------------------------------

_DEVOLUCION_ITEM_RE = re.compile(
    r'(\d{13})\n([^\n]+)\n(\d+)\n\d+\n\d+\n\d+',
)


def _parse_devolucion_items(raw_content: str) -> dict[str, tuple[str, int]]:
    """Extrae {isbn: (titulo, recibida)} del formato 'Diferencias en Devoluciones'.

    Formato por ítem en el texto crudo:
        {ISBN-13}\\n{Título}\\n{Recibida}\\n{Documentado}\\n{Fallada}\\n{Deteriorado}
    """
    result: dict[str, tuple[str, int]] = {}
    for isbn, titulo, recibida in _DEVOLUCION_ITEM_RE.findall(raw_content):
        result[isbn] = (titulo, int(recibida))
    return result


def _enrich_lines_from_devolucion(lines: list[DocumentLine], raw_content: str) -> None:
    """Completa ISBN y Cantidad en líneas de acuses cuando Azure no los extrajo (in-place).

    Busca en el texto crudo usando el patrón de devolución. Para cada línea sin
    product_code, intenta localizar el ISBN por coincidencia de título.
    """
    parsed = _parse_devolucion_items(raw_content)
    if not parsed:
        return

    title_lookup: dict[str, str] = {
        titulo.upper().strip(): isbn
        for isbn, (titulo, _) in parsed.items()
    }

    for line in lines:
        if line.values.get("product_code"):
            continue  # Azure ya lo extrajo, no tocar

        desc = (line.values.get("description") or "").upper().strip()
        isbn = title_lookup.get(desc)
        if isbn:
            _, recibida = parsed[isbn]
            line.values["product_code"] = isbn
            if not line.values.get("quantity"):
                line.values["quantity"] = recibida


def _parse_document_subtype(content: str) -> str | None:
    """Extrae el tipo de comprobante (Factura, Nota de Crédito, etc.) desde el texto crudo."""
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
    # "factura" solo si no está en frases del tipo "no válido como factura"
    if re.search(r'(?<!no v[a\u00e1]lido como )\bfactura\b', content_lower):
        return "Factura"
    return None


_COPY_MARKERS = re.compile(r'\b(DUPLICADO|TRIPLICADO|CUADRUPLICADO)\b', re.IGNORECASE)


def _original_pages_param(pdf_path: Path) -> str | None:
    """Detecta si el PDF contiene copias (DUPLICADO/TRIPLICADO) y devuelve
    el rango de páginas del original.

    Escanea el texto de cada página buscando la primera que contenga una
    marca de copia. Si la encuentra en la página N, devuelve "1-{N-1}".
    Si no hay marcas de copia, devuelve None (leer todo el PDF).

    Requiere pypdf. Si no está instalado, devuelve None (fallback seguro).
    """
    if _PdfReader is None:
        return None
    try:
        reader = _PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if _COPY_MARKERS.search(text):
                last_original = i - 1
                if last_original < 1:
                    # La marca está en la primera página misma — no truncar
                    logger.warning(
                        "Marca de copia detectada en página 1 de '%s'. "
                        "Se omite el truncado para no perder el original.",
                        pdf_path.name,
                    )
                    return None
                logger.info(
                    "Detectada marca de copia en página %d de '%s'. "
                    "Azure procesará solo páginas 1-%d.",
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
                "Faltan credenciales Azure. Configurar AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT y AZURE_DOCUMENT_INTELLIGENCE_KEY."
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

        # Determinar páginas a enviar: detección automática de copias tiene
        # prioridad sobre el valor fijo de configuración.
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

        # invoice_due_date: solo válida para facturas con pago a futuro.
        # Si el comprobante es de contado, Azure puede devolver la fecha de emisión
        # como DueDate (no hay vencimiento real). Se fuerza a null.
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
        lines = _extract_lines(fields.get("Items"))

        if _should_use_raw_parser(raw_content, lines):
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
