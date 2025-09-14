import os
import io
import pandas as pd
from pathlib import Path
from mistralai import Mistral


class FileManager:
    def __init__(self):
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        self.client = Mistral(api_key=mistral_api_key)

    def list_libraries(self):
        libraries = self.client.beta.libraries.list().data
        if not libraries:
            return "No libraries found for the current user."

        result = "Available Libraries:\nFormat: [Library Name] -> ID: [library_id] | Documents: [count]\n\n"
        for library in libraries:
            result += f"[{library.name}] -> ID: {library.id} | Documents: {library.nb_documents}\n"
        return result.strip()

    def list_documents(self, library_id):
        doc_list = self.client.beta.libraries.documents.list(library_id=library_id).data
        if not doc_list:
            return f"No documents found in library with ID: {library_id}"

        result = f"Documents in Library {library_id}:\nFormat: [Document Name] -> ID: [document_id] | Type: [extension]\n\n"
        for doc in doc_list:
            result += f"[{doc.name}] -> ID: {doc.id} | Type: {doc.extension}\n"
        return result.strip()

    def extract_text(self, library_id, document_id):
        extracted_text = self.client.beta.libraries.documents.text_content(
            library_id=library_id, document_id=document_id
        )
        return extracted_text.text

    def analyze_csv(self, library_id, document_id, separator=","):
        text_content = self.extract_text(library_id, document_id)
        df = pd.read_csv(io.StringIO(text_content), sep=separator)

        summary = [
            "CSV Analysis Summary:",
            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            "",
            "Columns:",
        ]

        for i, col in enumerate(df.columns, 1):
            summary.append(f"  {i}. {col} ({df[col].dtype})")

        summary.extend(["", "Data Types:"])
        for dtype in df.dtypes.value_counts().items():
            summary.append(f"  {dtype[0]}: {dtype[1]} columns")

        summary.extend(["", "Missing Values:"])
        missing = df.isnull().sum()
        if missing.sum() == 0:
            summary.append("  No missing values")
        else:
            for col, count in missing[missing > 0].items():
                summary.append(f"  {col}: {count} missing")

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            summary.extend(
                ["", "Numeric Summary:", df[numeric_cols].describe().to_string()]
            )

        summary.extend(["", "Sample Data (first 5 rows):", df.head().to_string()])
        return "\n".join(summary)

    def save_document(self, library_id, document_id):
        documents = self.client.beta.libraries.documents.list(
            library_id=library_id
        ).data
        document_name = None
        for doc in documents:
            if doc.id == document_id:
                document_name = doc.name
                break

        if not document_name:
            return f"Error: Document with ID {document_id} not found in library {library_id}"

        text_content = self.extract_text(library_id, document_id)
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        file_path = datasets_dir / document_name

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        return f"File saved at: datasets/{document_name}"

    def find_file(self, filename):
        libraries = self.client.beta.libraries.list().data

        for library in libraries:
            documents = self.client.beta.libraries.documents.list(
                library_id=library.id
            ).data
            for doc in documents:
                if doc.name == filename:
                    return {
                        "library_id": library.id,
                        "library_name": library.name,
                        "document_id": doc.id,
                        "document_name": doc.name,
                        "document_type": doc.extension,
                    }

        return None

    def list_datasets(self):
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            return "datasets/ directory doesn't exist"

        csv_files = list(datasets_dir.glob("*.csv"))
        if not csv_files:
            return "No CSV files found in datasets/ directory"

        result = ["Available datasets:"]
        for file_path in sorted(csv_files):
            file_size = file_path.stat().st_size
            size_str = (
                f"{file_size / 1024:.1f} KB"
                if file_size > 1024
                else f"{file_size} bytes"
            )

            try:
                df = pd.read_csv(file_path)
                info = f"({df.shape[0]} rows, {df.shape[1]} cols)"
            except:
                info = "(unable to parse)"

            result.append(f"• {file_path.name} - {size_str} {info}")

        return "\n".join(result)

    def delete_dataset(self, filename):
        datasets_dir = Path("datasets")
        file_path = datasets_dir / filename

        if not file_path.exists():
            return f"Dataset '{filename}' not found in datasets/ directory"

        try:
            file_path.unlink()
            return f"Dataset '{filename}' deleted successfully"
        except Exception as e:
            return f"Error deleting dataset '{filename}': {str(e)}"
