python -m venv venv
call venv\Scripts\activate
pip install -e .

set ROOT=data

mkdir %ROOT%\dataset\wavs
mkdir %ROOT%\output\wavs
mkdir %ROOT%\output\text
mkdir %ROOT%\processed\chunks
mkdir %ROOT%\processed\separated
mkdir %ROOT%\processed\transcripts
mkdir %ROOT%\temp\runs
mkdir %ROOT%\raw\audio