# system-design-for-genai-talk

Code samples from the talk "System Design for the GenAI Era"

## Getting Started
Install the required packages using `uv`:
```bash
uv sync
```

Then, create a `.env` file in the root directory of the project. 
The `.env` file should contain the variables mentioned in the `.env.example` file.

Finally, run the scripts with:
```bash
uv run --env-file .env python "01 - indexing/indexing.py" 
```
and 
```bash
uv run --env-file .env python "02 - query/query.py"
```
