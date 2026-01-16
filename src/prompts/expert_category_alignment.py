category_alignment_template = """You will be given [description of claims and expert category]

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
[Example 1]
[Example 2]
[Example 3]

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_politeness = """You will be given a set of claims that relate to why a politeness rating was given to an utterance, and a series of categories that an expert linguist would use to perform this type of politeness classification.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning.
"""