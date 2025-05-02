decomposition_template = """
You will be given a paragraph that explains XXX. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT:

OUTPUT:

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_politeness = """
You will be given a paragraph that explains why a certain level of politeness was attributed to an utterance. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. Do not omit hedging language (e.g., "seems to," "somewhat," "lacks overt markers") or subjective phrasing if present.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: This utterance is formal and professional, with no overtly rude language. The phrasing is neutral-to-polite, as it avoids accusatory or dismissive tones. The use of "I am copying them here" is transparent and non-confrontational, and "seem to constitute" softens any potential imposition by acknowledging some level of subjectivity. However, it lacks explicit politeness markers such as "please" or "thank you," which would elevate it to "extremely polite."

OUTPUT:
The utterance is formal and professional.
The utterance contains no overtly rude language.
The phrasing is neutral-to-polite because it avoids accusatory or dismissive tones.
The phrase "I am copying them here" is transparent and non-confrontational.
The phrase "seem to constitute" softens any potential imposition by acknowledging subjectivity.
The utterance lacks explicit politeness markers such as "please" or "thank you."
The lack of explicit politeness markers prevents the utterance from being considered "extremely polite."

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_emotion = """
You will be given a paragraph that explains why the given emotion was most reflected in a Reddit comment. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The use of "creepy" and "tbh" suggests a negative reaction or discomfort. Overall, this likely indicates the speaker finds the voicepack irritating or unsettling.

OUTPUT:
The use of "creepy" and "tbh" suggest a negative reaction.
The use of "creepy" and "tbh" suggests a feeling of discomfort.
The overall word choice likely indicates the speaker finds the voicepack irritating or unsettling.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""