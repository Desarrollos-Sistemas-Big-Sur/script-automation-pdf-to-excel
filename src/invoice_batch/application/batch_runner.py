from __future__ import annotations

import logging
from datetime import datetime

from invoice_batch.application.invoice_processor import InvoiceProcessor
from invoice_batch.config import AppConfig
from invoice_batch.domain.contracts import FileManager, Mailer, RunReporter
from invoice_batch.domain.models import RunContext, RunSummary


class BatchRunner:
    def __init__(
        self,
        config: AppConfig,
        processor: InvoiceProcessor,
        file_manager: FileManager,
        reporter: RunReporter,
        mailer: Mailer,
        logger: logging.Logger,
        onedrive_client=None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.file_manager = file_manager
        self.reporter = reporter
        self.mailer = mailer
        self.logger = logger
        self.onedrive_client = onedrive_client

    def run(self) -> RunSummary:
        started_at = datetime.now()
        run_id = started_at.strftime("%Y%m%d_%H%M%S")
        self.file_manager.ensure_directories()

        od_item_map: dict[str, str] = {}

        if self.onedrive_client is not None:
            self.logger.info("OneDrive habilitado — verificando carpetas...")
            self.onedrive_client.ensure_all_folders()

            pending = self.onedrive_client.list_pending()
            if not pending:
                self.logger.info("No hay PDFs pendientes en OneDrive.")
            else:
                self.logger.info("Descargando %d PDF(s) desde OneDrive Pendientes...", len(pending))
                inbox = self.config.paths.input_dir
                inbox.mkdir(parents=True, exist_ok=True)
                for item in pending:
                    local_file = inbox / item["name"]
                    try:
                        self.onedrive_client.download(item["id"], local_file)
                        od_item_map[item["name"]] = item["id"]
                        self.logger.info("Descargado: %s", item["name"])
                    except Exception as exc:
                        self.logger.error("No se pudo descargar %s: %s", item["name"], exc)

        input_files = self.file_manager.discover_pending_files()
        context = RunContext(run_id=run_id, started_at=started_at, input_files=input_files)

        self.reporter.report_start(context)

        results = []
        fatal_error = None

        try:
            for file_path in input_files:
                result = self.processor.process_file(context, file_path)
                results.append(result)
                self.reporter.report_file_result(result)

                if self.onedrive_client is not None and file_path.name in od_item_map:
                    item_id = od_item_map[file_path.name]
                    success = result.status in ("success", "warning")
                    needs_review = any(
                        msg.code == "total_amount_mismatch"
                        for msg in result.validation_messages
                    )
                    try:
                        self.onedrive_client.archive_pdf(
                            item_id,
                            file_path.name,
                            success,
                            needs_review=needs_review,
                        )
                    except Exception as exc:
                        self.logger.error(
                            "No se pudo archivar %s en OneDrive: %s", file_path.name, exc
                        )
        except Exception as exc:
            fatal_error = str(exc)
            self.logger.exception("Fallo fatal de corrida")

        subtype_counts: dict[str, int] = {}
        for item in results:
            if item.document is not None:
                subtype = item.document.fields.get("document_subtype") or "Desconocido"
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

        summary = RunSummary(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(),
            total_files=len(input_files),
            success_count=sum(1 for item in results if item.status == "success"),
            warning_count=sum(1 for item in results if item.status == "warning"),
            error_count=sum(1 for item in results if item.status == "error"),
            skipped_count=0,
            results=results,
            fatal_error=fatal_error,
            subtype_counts=subtype_counts,
        )

        self.reporter.report_finish(summary)
        self.processor.writer.finalize_run(summary)

        if self.config.reporting.email_enabled:
            self.mailer.send_run_summary(summary)

        return summary
