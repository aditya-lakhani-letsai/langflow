# src/backend/base/langflow/components/processing/prompt_parser.py
from loguru import logger
from langflow.custom import Component
from langflow.io import MessageInput, Output
from langflow.schema import Data
from langflow.schema.message import Message


class PromptParserComponent(Component):
    display_name = "LetsAI Prompt Parser"
    description = "Separates the prompt and the paths"
    icon = "message-square-share"
    beta = True
    name = "LetsaiPromptParser"

    inputs = [
        MessageInput(
            name="message",
            display_name="Message",
            info="The Message object to convert to a Data object",
        ),
    ]

    outputs = [
        Output(display_name="Prompt", name="prompt", method="get_prompt"),
        Output(display_name="Path", name="path", method="get_paths")  # Changed "file" to "path"
    ]

    def get_prompt(self) -> Message:
        """Extracts the prompt from the message."""
        if isinstance(self.message, Message):
            # Extracting the message text before the first pipe '|'
            message_text = self.message.text.split('|')[0].strip()  # Get text before first pipe
            return message_text
        return "No prompt found"

    def get_paths(self) -> Message:
        """Returns the first path as a Message object."""
        if isinstance(self.message, Message):
            paths_str = '|'.join(self.message.text.split('|')[1:]).strip()
            paths = [path.strip().strip('"') for path in paths_str.split('|') if path.strip()]
    
            if paths:
                return Message(text=json.dumps({"path": paths[0]}))
            else:
                return Message(text=json.dumps({"error": "No paths found"}))
        return Message(text=json.dumps({"error": "Invalid message"}))

    def get_file_paths(self) -> Data:
        """Generate file paths based on the file names."""
        if isinstance(self.message, Message):
            # Extract the file names and generate file paths
            paths_str = '|'.join(self.message.text.split('|')[1:]).strip()
            path_names = [path.strip() for path in paths_str.split('|') if path.strip()]

            # Generate file paths based on the path names
            file_paths = [f"/path/to/files/{path_name}" for path_name in path_names]
            return Data(data={"file_paths": file_paths})
        return Data(data={"error": "No file paths found"})
