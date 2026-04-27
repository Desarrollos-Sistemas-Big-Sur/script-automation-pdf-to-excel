from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class PathsConfig:
    input_dir: Path
    working_dir: Path
    processed_dir: Path
    error_dir: Path
    output_dir: Path
    log_dir: Path


@dataclass(slots=True)
class ProcessingConfig:
    allowed_document_types: list[str] = field(
        default_factory=lambda: ["invoice", "return_acknowledgement"]
    )
    supported_extensions: list[str] = field(default_factory=lambda: [".pdf"])
    continue_on_error: bool = True
    write_raw_json: bool = True
    output_strategy: str = "per_input_file"


@dataclass(slots=True)
class ReportingConfig:
    email_enabled: bool = False
    summary_recipients: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CsvConfig:
    per_document_columns: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationConfig:
    required_fields_by_document_type: dict[str, list[str]] = field(default_factory=dict)
    invoice_rules: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AzureConfig:
    endpoint: str
    key: str
    model_id: str = "prebuilt-invoice"
    pages: str | None = None


@dataclass(slots=True)
class OneDriveConfig:
    enabled: bool = False
    user_email: str = ""
    folder_path: str = ""
    pending_folder: str = "Pendientes"
    processed_folder: str = "Procesados"
    error_folder: str = "Errores"
    review_folder: str = "Revisar"   # facturas con discrepancia de totales


@dataclass(slots=True)
class GraphConfig:
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    sender: str = ""


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    processing: ProcessingConfig
    reporting: ReportingConfig
    csv: CsvConfig
    validation: ValidationConfig
    azure: AzureConfig
    graph: GraphConfig
    onedrive: OneDriveConfig


def _resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> AppConfig:
    load_dotenv()

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent.parent

    paths = PathsConfig(
        input_dir=_resolve_path(raw["paths"]["input_dir"], root),
        working_dir=_resolve_path(raw["paths"]["working_dir"], root),
        processed_dir=_resolve_path(raw["paths"]["processed_dir"], root),
        error_dir=_resolve_path(raw["paths"]["error_dir"], root),
        output_dir=_resolve_path(raw["paths"]["output_dir"], root),
        log_dir=_resolve_path(raw["paths"]["log_dir"], root),
    )
    processing = ProcessingConfig(**raw.get("processing", {}))
    reporting = ReportingConfig(**raw.get("reporting", {}))
    csv = CsvConfig(**raw.get("csv", {}))
    validation = ValidationConfig(**raw.get("validation", {}))

    azure_raw = raw.get("azure", {})
    azure = AzureConfig(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
        key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", ""),
        model_id=os.getenv("AZURE_DOCUMENT_MODEL_ID", "prebuilt-invoice"),
        pages=azure_raw.get("pages", None),
    )
    graph = GraphConfig(
        tenant_id=os.getenv("GRAPH_TENANT_ID", ""),
        client_id=os.getenv("GRAPH_CLIENT_ID", ""),
        client_secret=os.getenv("GRAPH_CLIENT_SECRET", ""),
        sender=os.getenv("GRAPH_SENDER", ""),
    )

    onedrive_raw = raw.get("onedrive", {})
    onedrive = OneDriveConfig(
        enabled=onedrive_raw.get("enabled", False),
        user_email=onedrive_raw.get("user_email", ""),
        folder_path=onedrive_raw.get("folder_path", ""),
        pending_folder=onedrive_raw.get("pending_folder", "Pendientes"),
        processed_folder=onedrive_raw.get("processed_folder", "Procesados"),
        error_folder=onedrive_raw.get("error_folder", "Errores"),
        review_folder=onedrive_raw.get("review_folder", "Revisar"),
    )

    return AppConfig(
        paths=paths,
        processing=processing,
        reporting=reporting,
        csv=csv,
        validation=validation,
        azure=azure,
        graph=graph,
        onedrive=onedrive,
    )
