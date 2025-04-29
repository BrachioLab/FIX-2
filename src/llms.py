import os
import concurrent.futures
from openai import OpenAI

class MyOpenAIModel:
    """
    A wrapper class for OpenAI's chat completion API that supports both single and batch processing.

    Usage:
        >>> llm = MyOpenAIModel("gpt-4o-mini")
        >>> out = llm(["What's your name?", "What's 1+1?"])
        >>> print(out)
        ['I'm called Assistant. How can I help you today?', '1 + 1 equals 2.']
    """
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = OpenAI(api_key=self.api_key)

    def call_openai(
        self,
        messages: list[dict],
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            stop=stop,
            seed=seed,
            response_format=response_format,
        )
        return response.choices[0].message.content.strip()

    def __call__(
        self,
        prompt: str | list[str],
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
        batch_size: int = 16,
    ) -> str | list[str]:
        # Convert single prompt to list for uniform processing
        is_single_prompt = isinstance(prompt, str)
        prompts = [prompt] if is_single_prompt else prompt
        
        # Process prompts in batches
        all_responses = []
        batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for batch in batches:
                futures = [
                    executor.submit(
                        self.call_openai,
                        messages=[{"role": "user", "content": p}],
                        temperature=temperature,
                        stop=stop,
                        response_format=response_format,
                        seed=seed,
                    )
                    for p in batch
                ]
                all_responses.extend(f.result() for f in futures)
        
        return all_responses[0] if is_single_prompt else all_responses