### This is a simple prototype using streamlit to use context from web url

For the purpose of this exercise I used OpenAI api so you will need to add an `.env` file with your api key.

```
OPENAI_API_KEY=<your_api_key>
```


You will want to create a virtual environment to run this:

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
pip install -r requirements.txt
```

To run the app:

```bash
streamlit run src/app.py
```

