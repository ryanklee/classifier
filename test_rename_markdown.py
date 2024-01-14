import sys
import os
import pytest
from unittest.mock import patch, mock_open, MagicMock

# Add the directory containing rename_markdown.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rename_markdown

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

# Function to put all files into a vector store and use llamaindex and chromadb
def put_files_into_vector_store_and_use_llamaindex_chromadb(directory):
    vector_store = ChromaIndex.ChromaIndex()
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                vector_store.add_document(content)
    return vector_store

# Test the process_markdown_files function
@patch('os.listdir')
@patch('os.rename')
@patch('builtins.open', new_callable=mock_open)
@patch('rename_markdown.get_suggested_title_and_filename')
def test_process_markdown_files(mock_get_suggested_title_and_filename, mock_open, mock_rename, mock_listdir):
    test_input_dir = "tests/test_data"
    test_output_dir = "tests/test_data_out"
    mock_listdir.return_value = ['note_1.md', 'note_2.md', 'note_3.md']
    mock_get_suggested_title_and_filename.side_effect = [
        ("My First Note", "my_first_note.md"),
        ("Recipe for Banana Bread", "recipe_for_banana_bread.md"),
        ("Meeting Minutes 2021-03-15", "meeting_minutes_2021_03_15.md")
    ]
    with patch('builtins.open', mock_open(read_data="# My First Note\n...")):
        rename_markdown.process_markdown_files(test_input_dir)
    mock_rename.assert_has_calls([
        call(os.path.join(test_input_dir, 'note_1.md'), os.path.join(test_output_dir, 'my_first_note.md')),
        call(os.path.join(test_input_dir, 'note_2.md'), os.path.join(test_output_dir, 'recipe_for_banana_bread.md')),
        call(os.path.join(test_input_dir, 'note_3.md'), os.path.join(test_output_dir, 'meeting_minutes_2021_03_15.md')),
    ], any_order=True)

# Test the put_files_into_vector_store_and_use_llamaindex_chromadb function
@patch('os.listdir')
@patch('builtins.open', new_callable=mock_open)
def test_put_files_into_vector_store_and_use_llamaindex_chromadb(mock_open, mock_listdir):
    test_directory = "tests/test_data"
    mock_listdir.return_value = ['note_1.md', 'note_2.md', 'note_3.md']
    mock_open.side_effect = [
        mock_open(read_data="Test content 1").return_value,
        mock_open(read_data="Test content 2").return_value,
        mock_open(read_data="Test content 3").return_value
    ]
    vector_store = put_files_into_vector_store_and_use_llamaindex_chromadb(test_directory)
    assert len(vector_store) == 3
    # Add more assertions as needed to verify the contents of the vector store

# Add more tests as needed to cover edge cases and error handling
