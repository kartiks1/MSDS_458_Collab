import json
from bs4 import BeautifulSoup
import re
import sys

def clean_markdown(text):
    # Remove special characters
    text = text.replace('\u00b6', '')
    
    # Convert headers
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', text)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', text)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', text)
    
    # Convert HTML links to markdown
    text = re.sub(r'<a href="([^"]+)">([^<]+)</a>', r'[\2](\1)', text)
    
    # Convert HTML formatting to markdown
    text = text.replace('<strong>', '**').replace('</strong>', '**')
    text = text.replace('<em>', '*').replace('</em>', '*')
    text = text.replace('<code>', '`').replace('</code>', '`')
    
    # Convert lists
    text = re.sub(r'<ul[^>]*>(.*?)</ul>', lambda m: '\n' + m.group(1) + '\n', text, flags=re.DOTALL)
    text = re.sub(r'<ol[^>]*>(.*?)</ol>', lambda m: '\n' + m.group(1) + '\n', text, flags=re.DOTALL)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'* \1\n', text)
    
    # Convert blockquotes
    text = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', lambda m: '> ' + m.group(1).replace('\n', '\n> '), text, flags=re.DOTALL)
    
    # Add proper line breaks
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', r'\n', text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def clean_code(text):
    # Remove "In [X]:" prefixes
    text = re.sub(r'In\s*\[\d+\]:\s*', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

def extract_outputs(cell):
    outputs = []
    output_div = cell.find('div', class_='output')
    if output_div:
        # Extract text output
        text_output = output_div.find('pre')
        if text_output:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [text_output.get_text()]
            })
        
        # Extract error output
        error_output = output_div.find('pre', class_='error')
        if error_output:
            outputs.append({
                "output_type": "error",
                "ename": "Error",
                "evalue": error_output.get_text(),
                "traceback": [error_output.get_text()]
            })
            
    return outputs

def convert_html_to_notebook(html_file, output_file):
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Find all cells
    cells = soup.find_all('div', class_='cell')
    
    for cell in cells:
        cell_type = 'code' if 'code_cell' in cell.get('class', []) else 'markdown'
        
        if cell_type == 'code':
            # Extract code
            code_div = cell.find('div', class_='input')
            if code_div:
                code = code_div.get_text().strip()
                code = clean_code(code)
                cell_data = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": extract_outputs(cell),
                    "source": [code]
                }
                notebook['cells'].append(cell_data)
        else:
            # Extract markdown
            markdown_div = cell.find('div', class_='text_cell_render')
            if markdown_div:
                markdown = markdown_div.decode_contents()  # Get HTML content
                markdown = clean_markdown(markdown)
                notebook['cells'].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [markdown]
                })

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_html_to_notebook.py <input_html_file> <output_ipynb_file>")
        sys.exit(1)
    
    html_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_html_to_notebook(html_file, output_file) 