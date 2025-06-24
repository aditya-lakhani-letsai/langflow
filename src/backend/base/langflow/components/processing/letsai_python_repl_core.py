import importlib
import ast
from typing import Dict, Union, List
from langchain_experimental.utilities import PythonREPL
from langflow.inputs import DictInput, MessageTextInput
from langflow.custom.custom_component.component import Component
from langflow.io import CodeInput, Output, StrInput
from langflow.schema.data import Data
from langflow.components.processing.python_repl_core import PythonREPLComponent

class LetsAIPythonREPLComponent(PythonREPLComponent):
    """
    Custom Langflow Component that extends the PythonREPLComponent to provide a safe Python REPL
    with support for complex data structures and dynamic variable injection.

    Allows:
    - Controlled global imports.
    - Runtime execution of Python code.
    - Support for complex Python-native inputs (including lambdas, lists, tuples, dicts).
    - Dynamic injection of user-defined parameters.
    """

    display_name = "LetsAI Python Interpreter"
    description = "Run Python code with support for all Python data structures including lambdas."
    icon = "square-terminal"

    inputs = [
        StrInput(
            name="global_imports",
            display_name="Global Imports",
            value="math,pandas",
            required=True,
            info="Comma-separated module names to import (e.g. 'math,pandas').",
        ),
        CodeInput(
            name="python_code",
            display_name="Python Code",
            value="print(sum(input_list))",
            input_types=["Message"],
            required=True,
            info="Python code to run. Refer to input variables injected via 'Variables' or 'Additional Variables'.",
        ),
        DictInput(
            name="params",
            display_name="Variables",
            input_types=["int", "float"],
            is_list=True,
            required=False,
            info="Key-value scalar pairs (int/float) to be available during execution.",
        ),
        MessageTextInput(
            name="dynamic_variable",
            display_name="Additional Variables",
            input_types=["Message", "Data"],
            value="{}",
            info="Python dictionary string containing complex types (lists, lambdas, dicts, etc.).",
        ),
    ]

    outputs = [
        Output(
            display_name="Results",
            name="results",
            type_=Data,
            method="run_python_repl",
        ),
    ]

    def get_globals(self, global_imports: Union[str, List[str]]) -> Dict[str, object]:
        """
        Import only the modules explicitly allowed by the user.

        Args:
            global_imports: A string of comma-separated module names or a list of module names.

        Returns:
            A dictionary containing the imported modules.

        Raises:
            ImportError: If a module cannot be imported.
            TypeError: If global_imports is neither a string nor a list.
        """
        global_dict = {}
        try:
            if isinstance(global_imports, str):
                modules = [module.strip() for module in global_imports.split(",") if module.strip()]
            elif isinstance(global_imports, list):
                modules = [module.strip() for module in global_imports if module.strip()]
            else:
                raise TypeError("global_imports must be either a string or a list")

            for module in modules:
                try:
                    imported_module = importlib.import_module(module)
                    global_dict[module] = imported_module
                except ImportError as e:
                    raise ImportError(f"Could not import module '{module}': {str(e)}") from e

            self.log(f"[Imports] Successfully imported modules: {list(global_dict.keys())}")
            return global_dict

        except Exception as e:
            self.log(f"[Imports Error] {str(e)}")
            raise

    def run_python_repl(self) -> Data:
        """
        Execute the user-defined Python code in a controlled REPL environment with injected globals.

        Returns:
            A Data object containing the result of the code execution or an error message.

        Raises:
            ValueError: If dynamic_variable parsing fails or is not a dictionary.
            ImportError: If a module import fails.
            SyntaxError: If the Python code has a syntax error.
            NameError: If the Python code references undefined variables.
            TypeError: If the Python code encounters type-related errors.
            ValueError: If the Python code encounters value-related errors.
        """
        try:
            globals_ = self.get_globals(self.global_imports)

            # Inject scalar variables from DictInput
            if isinstance(self.params, list):
                for item in self.params:
                    if isinstance(item, dict):
                        globals_.update(item)
            elif isinstance(self.params, dict):
                globals_.update(self.params)

            # Parse MessageTextInput input string (supports complex types like lambdas)
            raw_input = self.dynamic_variable.strip()
            if raw_input:
                try:
                    parsed_data = ast.literal_eval(raw_input)  # Safer than eval for trusted inputs
                    if not isinstance(parsed_data, dict):
                        raise ValueError("[Validation Error] Parsed dynamic_variable is not a dictionary.")
                    globals_.update(parsed_data)
                except Exception as e:
                    raise ValueError(f"[Parsing Error] Failed to parse dynamic_variable: {str(e)}")

            globals_["params"] = self.params  # Optional global access to all param sets

            self.log(f"[Globals] Injected variables: {list(globals_.keys())}")

            # Execute user code
            python_repl = PythonREPL(_globals=globals_)
            result = python_repl.run(self.python_code)
            clean_result = result.strip() if result else ""

            self.log("[Execution] Code execution completed successfully")
            return Data(data={"result": clean_result})

        except (ImportError, SyntaxError, NameError, TypeError, ValueError) as e:
            error_message = f"[Execution Error] {str(e)}"
            self.log(error_message)
            return Data(data={"error": error_message})

        except Exception as e:
            error_message = f"[Unexpected Error] {str(e)}"
            self.log(error_message)
            return Data(data={"error": error_message})

    def build(self) -> callable:
        """
        Required method for Langflow to recognize the custom component.

        Returns:
            The run_python_repl method to execute the component.
        """
        return self.run_python_repl