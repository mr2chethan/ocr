this repo helps to compare ocr engines

curl --location 'http://localhost:5001/easy-ocr' \
--form 'image=@"absolute/path/to/image"'

curl --location 'http://localhost:5001/paddle-ocr' \
--form 'image=@"absolute/path/to/image"'

curl --location 'http://localhost:5001/transformer-ocr' \
--form 'image=@"absolute/path/to/image"'

curl --location 'http://localhost:5001/tesseract-ocr' \
--form 'image=@"absolute/path/to/image"'
