from __future__ import annotations

import logging
import time
from pathlib import Path

from invoice_batch.config import AppConfig

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
_EXCEL_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Mapeo de document_subtype → nombre de subcarpeta en OneDrive
_SUBTYPE_TO_FOLDER: dict[str, str] = {
    "Factura": "Facturas",
    "Nota de Crédito": "Notas de Crédito",
    "Nota de Débito": "Notas de Débito",
    "Acuse de Devolución": "Acuses de Devolución",
}


def subfolder_for_subtype(document_subtype: str | None) -> str | None:
    """Devuelve el nombre de la subcarpeta de output según el tipo documental."""
    if not document_subtype:
        return None
    return _SUBTYPE_TO_FOLDER.get(document_subtype)


class OneDriveClient:
    def __init__(self, config: AppConfig) -> None:
        self.graph = config.graph
        self.od = config.onedrive
        self.logger = logging.getLogger("invoice_batch.onedrive")
        self._token: str | None = None
        self._token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Obtiene token OAuth2 via client credentials. Cachea hasta 60s antes de expirar."""
        if self._token and time.time() < self._token_expiry:
            return self._token
        resp = requests.post(
            _TOKEN_URL.format(tenant_id=self.graph.tenant_id),
            data={
                "grant_type": "client_credentials",
                "client_id": self.graph.client_id,
                "client_secret": self.graph.client_secret,
                "scope": "https://graph.microsoft.com/.default",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 60
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _item_url(self, path: str) -> str:
        """URL de Graph API para un item por ruta relativa al root del OneDrive."""
        return f"{_GRAPH_BASE}/users/{self.od.user_email}/drive/root:/{path}"

    # ------------------------------------------------------------------
    # Operaciones de carpeta
    # ------------------------------------------------------------------

    def ensure_folder(self, folder_path: str) -> str:
        """Crea la carpeta si no existe. Devuelve su ID."""
        parts = folder_path.strip("/").split("/")
        parent_id = "root"
        current_path = ""

        for part in parts:
            current_path = f"{current_path}/{part}".lstrip("/")
            resp = requests.get(
                self._item_url(current_path),
                headers=self._headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                parent_id = resp.json()["id"]
            elif resp.status_code == 404:
                create_resp = requests.post(
                    f"{_GRAPH_BASE}/users/{self.od.user_email}/drive/items/{parent_id}/children",
                    headers={**self._headers(), "Content-Type": "application/json"},
                    json={"name": part, "folder": {}},
                    timeout=15,
                )
                create_resp.raise_for_status()
                parent_id = create_resp.json()["id"]
            else:
                resp.raise_for_status()

        return parent_id

    def ensure_all_folders(self) -> None:
        """Garantiza que existan todas las carpetas operativas y de output."""
        base = self.od.folder_path
        folders = [
            f"{base}/{self.od.pending_folder}",
            f"{base}/{self.od.processed_folder}",
            f"{base}/{self.od.error_folder}",
            f"{base}/{self.od.review_folder}",
        ] + [f"{base}/{sub}" for sub in _SUBTYPE_TO_FOLDER.values()]

        for folder in folders:
            self.ensure_folder(folder)
            self.logger.debug("Carpeta verificada/creada: %s", folder)

    # ------------------------------------------------------------------
    # Listar y descargar pendientes
    # ------------------------------------------------------------------

    def list_pending(self) -> list[dict]:
        """Lista archivos PDF en la carpeta Pendientes. Devuelve lista de {name, id}."""
        pending_path = f"{self.od.folder_path}/{self.od.pending_folder}"
        resp = requests.get(
            f"{self._item_url(pending_path)}:/children",
            headers=self._headers(),
            params={"$select": "name,id,file", "$top": "200"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("value", [])
        return [
            {"name": i["name"], "id": i["id"]}
            for i in items
            if "file" in i and i["name"].lower().endswith(".pdf")
        ]

    def download(self, item_id: str, local_path: Path) -> None:
        """Descarga un archivo de OneDrive al disco local."""
        resp = requests.get(
            f"{_GRAPH_BASE}/users/{self.od.user_email}/drive/items/{item_id}/content",
            headers=self._headers(),
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()
        local_path.write_bytes(resp.content)

    # ------------------------------------------------------------------
    # Mover PDFs procesados
    # ------------------------------------------------------------------

    def _move_item(self, item_id: str, destination_folder_path: str) -> None:
        """Mueve un item a la carpeta destino (por path)."""
        dest_id = self.ensure_folder(destination_folder_path)
        resp = requests.patch(
            f"{_GRAPH_BASE}/users/{self.od.user_email}/drive/items/{item_id}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"parentReference": {"id": dest_id}},
            timeout=15,
        )
        resp.raise_for_status()

    def archive_pdf(
        self,
        item_id: str,
        filename: str,
        success: bool,
        needs_review: bool = False,
    ) -> None:
        """Mueve el PDF de Pendientes a la carpeta correspondiente.

        - Error al procesar              → Errores
        - Procesado con warnings de total → Revisar
        - Procesado correctamente         → Procesados
        """
        if not success:
            dest = f"{self.od.folder_path}/{self.od.error_folder}"
        elif needs_review:
            dest = f"{self.od.folder_path}/{self.od.review_folder}"
        else:
            dest = f"{self.od.folder_path}/{self.od.processed_folder}"

        self._move_item(item_id, dest)
        self.logger.info("PDF archivado en OneDrive: %s → %s", filename, dest)

    # ------------------------------------------------------------------
    # Subir Excel
    # ------------------------------------------------------------------

    def upload_excel(
        self,
        local_path: Path,
        document_subtype: str | None,
        needs_review: bool = False,
    ) -> str:
        """Sube un Excel al subfolder correspondiente según el tipo documental.

        Si needs_review=True, sube a la carpeta Revisar en lugar del subfolder normal.
        """
        if needs_review:
            remote_folder = f"{self.od.folder_path}/{self.od.review_folder}"
        else:
            sub = subfolder_for_subtype(document_subtype)
            remote_folder = (
                f"{self.od.folder_path}/{sub}" if sub else self.od.folder_path
            )

        remote_path = f"{remote_folder}/{local_path.name}"

        resp = requests.put(
            f"{self._item_url(remote_path)}:/content",
            headers={**self._headers(), "Content-Type": _EXCEL_CONTENT_TYPE},
            data=local_path.read_bytes(),
            timeout=60,
        )
        resp.raise_for_status()
        web_url = resp.json().get("webUrl", "")
        self.logger.info("Excel subido: %s → %s", local_path.name, web_url)
        return web_url
