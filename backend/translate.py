import cohere
from google.colab import userdata

# Load the API key
api_key = userdata.get("COHERE")
if api_key is None:
    raise ValueError("Please set the COHER secret in Colab secrets.")

# Initialize client
co = cohere.ClientV2(api_key)

def translate_to_english(text: str) -> str:
    """
    Translate text from any language into English using Cohere's translation model.
    """
    response = co.chat(
        model="command-a-translate-08-2025",
        messages=[
            {
                "role": "user",
                "content": f"Translate the following into English:\n\n{text}",
            }
        ],
        # Optional: you might set other params like temperature, etc.
    )
    # response.message.content is a list; pick first message
    translated = response.message.content[0].text.strip()
    return translated
