import openai
import os

def get_cot_steps_from_chatgpt(user_query, api_key=None, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    prompt = (
        "You are a geospatial reasoning assistant. "
        "Given a user query, break down the solution into a step-by-step chain of thought. "
        "Return the steps as a numbered list, one step per line.\n\n"
        f"User query: {user_query}\n\nChain of Thought:"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    content = response.choices[0].message.content
    steps = [line.strip(" .") for line in content.split("\n") if line.strip() and line.strip()[0].isdigit()]
    return steps
