from __future__ import annotations

import logging
import re

from invoice_batch.domain.models import ExtractedDocument, ValidationMessage

logger = logging.getLogger("invoice_batch.validation")

# Tolerancia para la validación del total: 1% del total de la factura.
_TOTAL_TOLERANCE_PCT = 0.01

# Patrón para detectar códigos/ISBNs válidos (9 a 13 dígitos numéricos)
_CODE_RE = re.compile(r'^\d{9,13}$')


class ConfigurableValidator:
    def __init__(
        self,
        required_fields_by_document_type: dict[str, list[str]],
        invoice_rules: dict[str, object] | None = None,
    ) -> None:
        self.required_fields_by_document_type = required_fields_by_document_type
        self.invoice_rules = invoice_rules or {}

    def validate(self, document: ExtractedDocument) -> list[ValidationMessage]:
        messages: list[ValidationMessage] = []
        required_fields = self.required_fields_by_document_type.get(
            document.document_type,
            [],
        )

        for field_name in required_fields:
            if document.fields.get(field_name) in (None, "", []):
                messages.append(
                    ValidationMessage(
                        level="warning",
                        code="missing_required_field",
                        message=f"Falta campo requerido: {field_name}",
                    )
                )

        if document.document_type == "invoice":
            messages.extend(self._validate_invoice_due_date(document))
            messages.extend(self._validate_total_amount(document))
            messages.extend(self._validate_line_fields(document))

        return messages

    def _validate_invoice_due_date(
        self,
        document: ExtractedDocument,
    ) -> list[ValidationMessage]:
        messages: list[ValidationMessage] = []
        invoice_due_date = document.fields.get("invoice_due_date")
        payment_terms = (document.fields.get("payment_terms") or "").strip().lower()

        if invoice_due_date not in (None, ""):
            return messages

        allowed_terms = {
            str(value).strip().lower()
            for value in self.invoice_rules.get(
                "allow_missing_invoice_due_date_when_payment_terms",
                [],
            )
        }
        _contado_base = {"contado", "contado inmediato", "contado sin intereses", "consignacion", "consignación"}
        if payment_terms and (payment_terms in allowed_terms or payment_terms in _contado_base):
            return messages

        policy = self.invoice_rules.get(
            "missing_invoice_due_date_policy_for_other_payment_terms",
            "configurable",
        )
        if policy == "warning":
            messages.append(
                ValidationMessage(
                    level="warning",
                    code="missing_invoice_due_date",
                    message=(
                        "La factura no informa fecha_de_vencimiento_factura. "
                        "No se reemplaza con fecha_de_vencimiento_cae."
                    ),
                )
            )

        return messages

    def _validate_total_amount(
        self,
        document: ExtractedDocument,
    ) -> list[ValidationMessage]:
        """Valida que la suma de totales de línea coincida con el total de la factura."""
        total_amount = document.fields.get("total_amount")
        if total_amount is None:
            logger.debug("Validación de total omitida: total_amount no disponible.")
            return []

        line_totals = [
            line.values.get("line_total")
            for line in document.lines
            if line.values.get("line_total") is not None
        ]

        if not line_totals:
            logger.debug("Validación de total omitida: ninguna línea tiene line_total.")
            return []

        suma_lineas = sum(line_totals)
        tolerancia = abs(total_amount) * _TOTAL_TOLERANCE_PCT
        diferencia = abs(suma_lineas - total_amount)

        logger.info(
            "Validación de total — factura: %.2f | suma líneas: %.2f | diferencia: %.2f | tolerancia: %.2f",
            total_amount, suma_lineas, diferencia, tolerancia,
        )

        if diferencia > tolerancia:
            return [
                ValidationMessage(
                    level="warning",
                    code="total_amount_mismatch",
                    message=(
                        f"La suma de los totales de línea ({suma_lineas:,.2f}) "
                        f"no coincide con el total de la factura ({total_amount:,.2f}). "
                        f"Diferencia: {diferencia:,.2f}. "
                        "Revisar si faltan ítems."
                    ),
                )
            ]

        return []

    def _validate_line_fields(
        self,
        document: ExtractedDocument,
    ) -> list[ValidationMessage]:
        """Valida que cada línea tenga los campos mínimos necesarios.

        Una línea es inválida si:
          - product_code e isbn ambos ausentes (con uno alcanza)
          - unit_price ausente
          - line_discount ausente

        Si hay al menos una línea inválida, la factura va a Revisar.
        """
        if not document.lines:
            return []

        invalid_lines = []

        for line in document.lines:
            v = line.values

            # Verificar identificador: alcanza con product_code O isbn válido
            code = str(v.get("product_code") or "").strip()
            isbn = str(v.get("isbn") or "").strip()
            # product_code válido: 9-13 dígitos numéricos
            has_identifier = bool(
                _CODE_RE.match(code) or _CODE_RE.match(isbn)
            )

            has_price = v.get("unit_price") is not None
            has_discount = v.get("line_discount") is not None

            if not has_identifier or not has_price or not has_discount:
                invalid_lines.append(line.line_number)

        if invalid_lines:
            sample = invalid_lines[:5]
            more = len(invalid_lines) - len(sample)
            detail = ", ".join(str(n) for n in sample)
            if more > 0:
                detail += f" y {more} más"

            logger.warning(
                "Campos incompletos en %d línea(s): %s.",
                len(invalid_lines), detail,
            )

            return [
                ValidationMessage(
                    level="warning",
                    code="incomplete_line_fields",
                    message=(
                        f"{len(invalid_lines)} línea(s) sin campos obligatorios "
                        f"(identificador, precio o descuento). "
                        f"Líneas afectadas: {detail}."
                    ),
                )
            ]

        return []
