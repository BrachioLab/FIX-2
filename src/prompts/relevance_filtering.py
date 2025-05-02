relevance_template = """You will be given [description of input, output, and claim]

A claim is relevant if and only if:
(1) It is supported by the content of the input (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why XXX.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
[Example 2]
[Example 3]

Now, determine whether the following claim is relevant to the given XXX:
Input: {}
Output: {}
Claim: {}
"""

relevance_politeness = """You will be given an utterance, its politeness rating on a 1-5 scale (where 1: very rude and 5: very polite), and a claim that may or may not be relevant to an explanation of the rating. Your task is to decide whether the claim is relevant to explaining the politeness rating for this specific utterance.

A claim is relevant if and only if:
(1) It is supported by the content of the utterance (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why the utterance received the given politeness rating (i.e., it directly relates to tone, phrasing, or other aspects relevant to the rating).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Utterance: "There is no such fact - you are just making things up. There is no reason to believe that any person reading about Bologna would be particularly interested in Kappa Sigma. If they wanted to know about Kappa Sigma, they would read the Kappa Sigma article instead.."
Politeness Rating: 2
Claim: The utterance accuses the other person of fabricating information.
Relevance: Yes
Reasoning: The claim is relevant because it discusses the accusatory tone of the utterance, which contributes to its rudeness classification.

[Example 2]
Utterance: "Deleted reference to REM sleep in the first sentence. It simply is not true. In fact, REM deprivation is a common side effect of antidepressant use (some attribute their effects to REM deprivation)."
Politeness Rating: 3
Claim: The utterance is neutral.
Relevance: No
Reasoning: The claim is not relevant because it simply states the rating, and does not provide information about why the rating was given. Claims that merely state the rating are not relevant.

[Example 3]
Utterance: "Tetra-gram is a compound word as is the penta-gram. Penta refers to the number 5 in Greek, tetra refers to the number 4 and gram refers to the word line in both cases. Obviously a star shape can't be shaped with 4 lines."
Politeness Rating: 3
Claim: The use of "obviously" might suggest the author is an expert in Greek.
Relevance: No
Reasoning: The claim is not relevant because there is nothing in the text to support that the author may be Greek. 

Now, determine whether the following claim is relevant to the given utterance and politeness rating:
Utterance: {}
Politeness Rating: {}
Claim: {}
"""


relevance_cholec = """
You will be given:
- An endoscopic image of the gallbladder region during a laparoscopic cholecystectomy (the "Input").
- A textual Claim describing a proposed "safe" or "unsafe" zone in that image (the "Claim").

A claim is relevant if and only if:
1. It refers to a visually detectable feature in the image.
2. It pertains to identifying safe or unsafe dissection zones based on expert surgical criteria.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation pointing to the visual feature and criterion that supports your judgment>

Examples:

Example 1
Input: [image showing fully cleared Calot's triangle]
Claim: "Calot's triangle cleared."
```
Relevance: Yes
Reasoning: The claim describes the clearance of Calot's triangle, which is a visually confirmed safety criterion for safe dissection.
```

Example 2
Input: [image showing mottled gallbladder appearance]
Claim: "The gallbladder appears mottled green."
```
Relevance: No
Reasoning: While visible, the gallbladder color does not inform safe or unsafe dissection zones.
```

Now evaluate the following:
Input: (see attached image)
Claim: {}
"""

relevance_emotion = """You will be given a Reddit comment, its emotion label, and a claim that may or may not be relevant to an explanation of the emotion label. Your task is to decide whether the claim is relevant to explaining the emotion label for the specific text comment.

A claim is relevant if and only if:
(1) It is supported by the content of the utterance (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why the utterance received the given emotion label (i.e., it directly relates to tone, phrasing, sentiment, or other aspects relevant to the label).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Text: Apologies, I take it all back as I’ve just seen his latest effort
Emotion Label: remorse
Claim: The speaker shows regret and self‑reproach
Relevance: Yes
Reasoning: The speaker is explicitly apologizing and retracting their previous statement, which directly indicates remorse. The claim about the speaker showing regret and self-reproach aligns with the tone and content of the utterance.

[Example 2]
Text: At least it’s not anything worse, and that you are still close to that person :)
Label: relief
Claim: The speaker is expressing happiness at a positive outcome.
Relevance: No
Reasoning: This claim is not relevant as it does not relate to relief. The claim speculates a different emotion (happiness) that doesn't directly relate to the sentiment or tone of the given text.

[Example 3]
Text: seriously wtf. I want to see how the whole hand went in detail. that was the sickest soulread ever
Label: admiration
Claim: The exclamations (“seriously wtf… sickest soulread ever”) show astonished praise for an impressive play.
Relevance: Yes
Reasoning: The exclamations "seriously wtf" and "sickest soulread ever" express strong admiration and astonishment, which directly supports the emotion label of admiration. The claim accurately describes the tone of praise and amazement present in the utterance.

Now, determine whether the following claim is relevant to the given text and emotion label:
Text: {}
Emotion Label: {}
Claim: {}
"""