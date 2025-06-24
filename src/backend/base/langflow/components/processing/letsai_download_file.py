from uuid import UUID
from io import BytesIO
import json
from collections.abc import AsyncIterator, Iterator

import orjson
import pandas as pd
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder

from langflow.api.v2.files import upload_user_file
from langflow.custom import Component
from langflow.io import DropdownInput, HandleInput, StrInput
from langflow.schema import Data, DataFrame, Message
from langflow.services.auth.utils import create_user_longterm_token
from langflow.services.database.models.user.crud import get_user_by_id
from langflow.services.deps import get_session, get_settings_service, get_storage_service
from langflow.template.field.base import Output
import markdown
from weasyprint import HTML
from docx import Document

# Required for PDF and DOCX support
try:
    import markdown
    from weasyprint import HTML
    from docx import Document
except ImportError as e:
    raise ImportError(
        "PDF/DOCX export requires 'markdown', 'weasyprint', and 'python-docx'. Please install them using:\n"
        "pip install markdown weasyprint python-docx"
    ) from e


class SaveToFileComponent(Component):
    display_name = "LetsAI File Download"
    description = "Generate a downloadable file from input (in memory, no disk write)."
    icon = "download"
    name = "LetsaiDownloadFile"

    DATA_FORMAT_CHOICES = ["csv", "excel", "json", "markdown"]
    MESSAGE_FORMAT_CHOICES = ["txt", "json", "markdown", "pdf", "docx"]

    inputs = [
        HandleInput(
            name="input",
            display_name="Input",
            info="The input to convert into downloadable file.",
            dynamic=True,
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            info="Name of the file (without extension).",
            required=True,
        ),
        DropdownInput(
            name="file_format",
            display_name="File Format",
            options=list(dict.fromkeys(DATA_FORMAT_CHOICES + MESSAGE_FORMAT_CHOICES)),
            info="Select the file format. Defaults based on input type.",
            value="",
        ),
    ]

    outputs = [Output(display_name="Download Link", name="result", method="save_to_file")]

    async def save_to_file(self) -> Message:
        if not self.file_name:
            raise ValueError("File name must be provided.")

        input_type = self._get_input_type()
        file_format = self.file_format or self._get_default_format()

        allowed_formats = (
            self.MESSAGE_FORMAT_CHOICES if input_type == "Message" else self.DATA_FORMAT_CHOICES
        )
        if file_format not in allowed_formats:
            raise ValueError(f"Invalid file format '{file_format}' for {input_type}. Allowed: {allowed_formats}")

        file_bytes, filename = await self._generate_file_content(self.input, input_type, file_format)
        upload_file = UploadFile(filename=filename, file=BytesIO(file_bytes), size=len(file_bytes))
        file_id = await self._upload_in_memory_file(upload_file)

        return Message(text=f"[Click here to download](/api/v2/files/{file_id})")

    def _get_input_type(self) -> str:
        if type(self.input) is DataFrame:
            return "DataFrame"
        if type(self.input) is Message:
            return "Message"
        if type(self.input) is Data:
            return "Data"
        raise ValueError(f"Unsupported input type: {type(self.input)}")

    def _get_default_format(self) -> str:
        input_type = self._get_input_type()
        if input_type == "DataFrame":
            return "csv"
        if input_type == "Data":
            return "json"
        if input_type == "Message":
            return "json"
        return "json"

    async def _generate_file_content(self, input_data, input_type: str, fmt: str) -> tuple[bytes, str]:
        filename = f"{self.file_name}.{fmt}"
        if input_type == "DataFrame":
            return self._generate_from_dataframe(input_data, fmt), filename
        elif input_type == "Data":
            return self._generate_from_data(input_data, fmt), filename
        elif input_type == "Message":
            return await self._generate_from_message(input_data, fmt), filename
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def _generate_from_dataframe(self, df: DataFrame, fmt: str) -> bytes:
        buffer = BytesIO()
        if fmt == "csv":
            df.to_csv(buffer, index=False)
        elif fmt == "excel":
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
        elif fmt == "json":
            buffer.write(df.to_json(orient="records", indent=2).encode("utf-8"))
        elif fmt == "markdown":
            buffer.write(df.to_markdown(index=False).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        buffer.seek(0)
        return buffer.read()

    def _generate_from_data(self, data: Data, fmt: str) -> bytes:
        df = pd.DataFrame(data.data)
        return self._generate_from_dataframe(df, fmt)

    async def _generate_from_message(self, msg: Message, fmt: str) -> bytes:
        content = ""
        if msg.text is None:
            content = ""
        elif isinstance(msg.text, AsyncIterator):
            async for item in msg.text:
                content += str(item) + " "
        elif isinstance(msg.text, Iterator):
            content = " ".join(str(item) for item in msg.text)
        else:
            content = str(msg.text)

        buffer = BytesIO()
        if fmt == "txt":
            buffer.write(content.encode("utf-8"))
        elif fmt == "json":
            buffer.write(json.dumps({"message": content}, indent=2).encode("utf-8"))
        elif fmt == "markdown":
            buffer.write(f"\n\n{content}".encode("utf-8"))
        elif fmt == "pdf":
            html_content = markdown.markdown(content)
            pdf_bytes = HTML(string=html_content).write_pdf()
            buffer.write(pdf_bytes)
        elif fmt == "docx":
            document = Document()
            document.add_paragraph(content)
            document.save(buffer)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        buffer.seek(0)
        return buffer.read()

    async def _upload_in_memory_file(self, upload_file: UploadFile) -> UUID:
        async for db in get_session():
            user_id, _ = await create_user_longterm_token(db)
            current_user = await get_user_by_id(db, user_id)

            upload_response = await upload_user_file(
                file=upload_file,
                session=db,
                current_user=current_user,
                storage_service=get_storage_service(),
                settings_service=get_settings_service(),
            )
            return upload_response.id
