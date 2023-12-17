import pytest
import rename_markdown
from unittest.mock import patch, mock_open, MagicMock

# Test the sanitize_filename function
def test_sanitize_filename():
    assert rename_markdown.sanitize_filename("Valid-Name.md") == "Valid-Name.md"
    assert rename_markdown.sanitize_filename("Invalid/Name?.md") == "InvalidName.md"
    assert rename_markdown.sanitize_filename("Another:Invalid*Name<>.md") == "AnotherInvalidName.md"

# Test the get_suggested_title_and_filename function with a successful LLM response
@patch('subprocess.run')
def test_get_suggested_title_and_filename_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout='{"title": "Test Title", "filename": "test_title.md"}')
    title, filename = rename_markdown.get_suggested_title_and_filename("Test content")
    assert title == "Test Title"
    assert filename == "test_title.md"

# Test the get_suggested_title_and_filename function with a failed LLM response
@patch('subprocess.run')
def test_get_suggested_title_and_filename_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1)
    with pytest.raises(Exception):
        rename_markdown.get_suggested_title_and_filename("Test content")

# Test the process_markdown_files function
@patch('os.listdir')
@patch('os.rename')
@patch('builtins.open', new_callable=mock_open, read_data="Test content")
@patch('rename_markdown.get_suggested_title_and_filename')
def test_process_markdown_files(mock_get_suggested_title_and_filename, mock_open, mock_rename, mock_listdir):
    mock_listdir.return_value = ['test.md']
    mock_get_suggested_title_and_filename.return_value = ("Test Title", "test_title.md")
    rename_markdown.process_markdown_files("test_directory")
    mock_rename.assert_called_once_with("test_directory/test.md", "test_directory/test_title.md")

# Add more tests as needed to cover edge cases and error handling
