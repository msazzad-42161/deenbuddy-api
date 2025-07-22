# DeenBuddy API


This project is a RESTful API backend for Islamic resources, implemented in Python. It provides endpoints for accessing Quranic content (including surahs, ayahs, translations, and tafseer), and includes modules for loading Quran and tafseer data. The codebase uses FastAPI as the web framework and supports environment-based configuration. There are also test scripts and a requirements file for dependencies. The project structure is flat, with main logic and data loaders in separate Python files. No advanced features (such as authentication, rate limiting, or database models) are present in the codebase as provided.
DeenBuddy API is a RESTful backend built with Python and FastAPI. It provides endpoints for Islamic resources such as Quranic verses, prayer times, and Hadith collections. The API supports user authentication, bookmarking, and includes OpenAPI documentation.

## Features

- Quran endpoints: surahs, ayahs, translations, tafsir
- A buddy like chatbot feel

## Tech Stack

- Python 3.10+
- FastAPI
- ChromaDB
- Pydantic
- Uvicorn

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/msazzad-42161/deenbuddy-api.git
cd deenbuddy-api
mkvirtualenv env_name
workon env_name
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and update variables.

### Running the API

```bash
python main.py
```
### Testing the API

After running the API from the above code ☝️, run this 👇

```bash
python test_deenbuddy_api.py
```

Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

```
deenbuddy-api/
├── islamic_knowledge_db/
├── .env
├── .gitignore
├── main.py
├── quran_loader.py
├── README.md
├── requirements.txt
├── tafseer_loader.py
└── test_deenbuddy_api.py
```

## Contributing

- Fork the repo
- Create a feature branch
- Commit and push changes
- Open a pull request

## License

MIT License

## Contact

Open an issue or email [sazzadulislam944@gmail.com](mailto:sazzadulislam944@gmail.com).
