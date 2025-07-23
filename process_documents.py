import argparse
import json
import pathlib
import re
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel

class Invoice(BaseModel):
    """Pydantic model for invoice data structure."""
    company_name: str
    description: str
    issue_date: str


class InvoiceProcessor:
    """Main class for processing invoice files."""

    def __init__(self, data_folder: str) -> None:
        """
        Initialize the invoice processor.

        Args:
            data_folder: Path to the folder containing invoice files
        """
        self.data_folder = pathlib.Path(data_folder)
        self.client = genai.Client()
        self.current_year = datetime.now().year

    def validate_folder(self) -> bool:
        """
        Validate that the specified folder exists and is accessible.

        Returns:
            True if folder is valid, False otherwise
        """
        if not self.data_folder.exists():
            logger.error(f"Folder does not exist: {self.data_folder}")
            return False

        if not self.data_folder.is_dir():
            logger.error(f"Path is not a directory: {self.data_folder}")
            return False

        return True

    def get_pdf_files(self) -> List[pathlib.Path]:
        """
        Get all PDF files from the specified folder.

        Returns:
            List of pathlib.Path objects for PDF files
        """
        try:
            pdf_files = [
                file for file in self.data_folder.iterdir()
                if file.is_file() and file.suffix.lower() == '.pdf'
            ]
            logger.info(f"Found {len(pdf_files)} PDF files in {self.data_folder}")
            return pdf_files
        except PermissionError:
            logger.error(f"Permission denied accessing folder: {self.data_folder}")
            return []

    def sanitize_filename(self, text: str) -> str:
        """
        Remove or replace characters that are not allowed in filenames.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text safe for use in filenames
        """
        # Remove invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
        # Replace whitespace with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(' .')
        return sanitized

    def parse_date_to_yyyymmdd(self, date_str: str) -> str:
        """
        Convert dd-mm-yyyy format to YYYYMMDD format.

        Args:
            date_str: Date string in dd-mm-yyyy format

        Returns:
            Date string in YYYYMMDD format
        """
        try:
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            return date_obj.strftime("%Y%m%d")
        except ValueError:
            logger.warning(f"Could not parse date: {date_str}, using sanitized version")
            # Remove dashes and other non-numeric characters as fallback
            return re.sub(r'\D', '', date_str)

    def extract_invoice_data(self, filepath: pathlib.Path) -> Optional[Invoice]:
        """
        Extract invoice data using Google's Gemini API.

        Args:
            filepath: Path to the PDF file

        Returns:
            Invoice object if successful, None otherwise
        """
        prompt = (
            f"From this invoice retrieve the following information: "
            f"company name, one word description (for instance 'subscription', "
            f"'course', 'insurance'), and issue date in dd-mm-yyyy format. "
            f"If the year is not present in the invoice, assume it is {self.current_year}. "
            f"Make sure the description is in English"
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type='application/pdf',
                    ),
                    prompt
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Invoice,
                }
            )

            logger.debug(f"API response for {filepath.name}: {response.text}")

            # Parse the JSON response
            invoice_data = json.loads(response.text)
            return Invoice(**invoice_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {filepath.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting data from {filepath.name}: {e}")
            return None

    def generate_new_filename(self, invoice: Invoice, original_path: pathlib.Path) -> str:
        """
        Generate a new filename based on invoice data.

        Args:
            invoice: Invoice data object
            original_path: Original file path

        Returns:
            New filename string
        """
        company_name = self.sanitize_filename(invoice.company_name)
        description = self.sanitize_filename(invoice.description)
        issue_date = self.parse_date_to_yyyymmdd(invoice.issue_date)

        file_extension = original_path.suffix
        return f"{company_name}_{description}_{issue_date}{file_extension}"

    def rename_file(self, original_path: pathlib.Path, new_filename: str) -> bool:
        """
        Rename a file to the new filename.

        Args:
            original_path: Original file path
            new_filename: New filename

        Returns:
            True if successful, False otherwise
        """
        new_path = original_path.parent / new_filename

        # Check if target file already exists
        if new_path.exists():
            logger.warning(f"Target file already exists, skipping: {new_filename}")
            return False

        try:
            original_path.rename(new_path)
            logger.info(f"Renamed {original_path.name} to {new_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to rename {original_path.name}: {e}")
            return False

    def process_file(self, filepath: pathlib.Path) -> bool:
        """
        Process a single invoice file.

        Args:
            filepath: Path to the file to process

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing {filepath.name}")

        # Extract invoice data
        invoice = self.extract_invoice_data(filepath)
        if not invoice:
            return False

        # Generate new filename
        new_filename = self.generate_new_filename(invoice, filepath)

        # Rename the file
        return self.rename_file(filepath, new_filename)

    def process_all_files(self) -> Tuple[int, int]:
        """
        Process all PDF files in the specified folder.

        Returns:
            Tuple of (successful_count, total_count)
        """
        if not self.validate_folder():
            return 0, 0

        pdf_files = self.get_pdf_files()
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return 0, 0

        successful_count = 0

        for pdf_file in pdf_files:
            try:
                if self.process_file(pdf_file):
                    successful_count += 1
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {pdf_file.name}: {e}")
                continue

        return successful_count, len(pdf_files)


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Process invoice PDFs and rename them based on extracted data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/invoices
  %(prog)s ./sample_data --verbose
  %(prog)s ~/Documents/invoices --verbose
        """
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing invoice PDF files"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        processor = InvoiceProcessor(args.folder)
        successful, total = processor.process_all_files()

        if total == 0:
            logger.error("No files were processed")
            return 1

        logger.info(f"Processing completed: {successful}/{total} files processed successfully")

        if successful == total:
            logger.success("All files processed successfully!")
            return 0
        else:
            logger.warning(f"Some files failed to process ({total - successful} failures)")
            return 1

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
