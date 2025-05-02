alignment_template = """You will be given [description of claim and expert category]

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert categories:
[list of categories and their descriptions]
-----

Here are some examples:
[Example 1]
[Example 2]
[Example 3]

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""


alignment_politeness = """You will be given a single claim that relates to why a politeness rating was given to an utterance. You will also be given a series of categories that an expert linguist would use to perform this type of politeness classification.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert politeness categories:
1. Honorifics and Formal Address: The presence of respectful or formal address forms (e.g., “sir,” “usted,” “您”) signals politeness by expressing deference to the hearer’s status or social distance.
2. Courteous Politeness Markers: Words such as “please,” “kindly,” or their multilingual variants soften requests and reflect courteous intent.
3. Gratitude Expressions: Use of expressions like “thank you,” “thanks,” or “I appreciate it” signals recognition of the other’s contribution and positive face.
4. Apologies and Acknowledgment of Fault: Phrases such as “sorry” or “I apologize” express humility and repair social breaches, marking a clear politeness strategy.
5. Indirect and Modal Requests: Requests using modal verbs (“could you,” “would you”) or softening cues like “by the way” reduce imposition and signal respect for the hearer’s autonomy.
6. Hedging and Tentative Language: Words like “I think,” “maybe,” or “usually” lower assertion strength and make statements more negotiable, reflecting interpersonal sensitivity.
7. Inclusive Pronouns and Group-Oriented Phrasing: Use of “we,” “our,” or “together” expresses solidarity and reduces hierarchical distance in requests or critiques.
8. Greeting and Interaction Initiation: Opening with a salutation (“hi,” “hello”) creates a cooperative tone and frames the conversation positively.
9. Compliments and Praise: Positive evaluations (“great,” “awesome,” “neat”) attend to the hearer’s positive face and foster a friendly environment.
10. Softened Disagreement or Face-Saving Critique: When disagreeing, the use of softeners, partial agreements, or concern for clarity preserves the hearer’s dignity.
11. Urgency or Immediacy of Language: Utterances emphasizing emergency or speed (“asap,” “immediately”) can heighten perceived imposition and reduce politeness if not softened.
12. Avoidance of Profanity or Negative Emotion: The presence of strong negative words or swearing is a key indicator of rudeness and face threat.
13. Bluntness and Direct Commands: Requests lacking modal verbs or mitigation (“Do this”) are perceived as less polite due to their imperative structure.
14. Empathy or Emotional Support: Recognizing the hearer’s emotional context or challenges is a politeness strategy of concern and goodwill.
15. First-Person Subjectivity Markers: Statements that begin with “I think,” “I feel,” or “In my view” convey humility and subjectivity, reducing imposition.
16. Second Person Responsibility or Engagement: Sentences starting with “you” or directly addressing the hearer can either signal engagement or come across as accusatory, depending on context and tone.
17. Questions as Indirect Strategies: Questions (“what do you think?” or “could you clarify?”) reduce imposition by inviting rather than demanding input.
18. Discourse Management with Markers: Use of discourse markers like “so,” “then,” “but” organizes conversation flow and may help manage face needs in conflict or negotiation.
19. Ingroup Language and Informality: Use of group-identifying slang or casual expressions (“mate,” “dude,” “bro”) may foster solidarity or seem disrespectful, depending on relational norms.
-----

Here are some examples:
[Example 1]
Claim: The utterance accuses the other person of fabricating information.
Category: Second Person Responsibility or Engagement
Category Alignment Rating: 0.9
Reasoning: The claim centers on the utterance accusing the other person of fabricating information, which aligns strongly with the use of direct second-person language (“you are just making things up”). This construction directly targets the listener and assigns blame, which is a key aspect of the Second Person Responsibility or Engagement category. 

[Example 2]
Claim: The sentence structure is overly complex and difficult to follow.
Category: Discourse Management with Markers
Category Alignment Rating: 0.1
Reasoning: The claim concerns sentence complexity and readability, which is weakly aligned with how Discourse Management with Markers functions—lack of these markers impact information organization or textual flow. However, this category has low alignment because the claim is more about syntactic complexity than interpersonal politeness strategies. No other category more directly addresses sentence structure, so this is the closest fit, but with very low alignment.

[Example 3]
Claim: The use of "obviously" might suggest a slight assumption of common knowledge.
Category: Hedging and Tentative Language
Category Alignment Rating: 0.6
Reasoning: The word "obviously" implies a presumption that the information should be universally known or accepted, which can reduce the speaker's humility and increase the assertiveness of the statement. This contrasts with hedging strategies that usually aim to soften claims and maintain interpersonal sensitivity. While "obviously" itself is not a hedge, its use relates to the degree of assertiveness in an utterance—making this category moderately relevant. The alignment is not perfect because “obviously” actually opposes hedging, but the connection lies in the shared focus on epistemic stance and assertion strength.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""


alignment_cholec = """You will be given:
- A textual Claim describing a proposed "safe" or "unsafe" zone in laparoscopic cholecystectomy (the "Claim").

Your task is as follows:
1. Determine which expert safety criterion is most aligned with the Claim.
2. Rate how strongly the criterion aligns with the Claim on a scale of 0.0-1.0 (in increments of 0.1), where 0.0 means "not at all" and 1.0 means "perfectly."

Return your answer exactly in this format:
```
Category: <name of selected criterion>
Criterion Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the criterion and how you judged the alignment rating>
```

-----
Expert Safety Criteria:
1. Calot's triangle cleared - Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
2. Cystic plate exposed - The lower third of the gallbladder must be dissected off the liver to reveal the shiny cystic plate and ensure the correct dissection plane.
3. Only two structures visible - Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
4. Above the R4U line - Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to liver segment IV to avoid the common bile duct.
5. Infundibulum start point - Dissection should begin at the gallbladder infundibulum-cystic duct junction to stay in safe tissue planes.
6. Subserosal plane stay - When separating the gallbladder from the liver, stay in the avascular subserosal cleavage plane under the serosal fat layer.
7. Cystic lymph node guide - Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.
8. No division without ID - Never divide any duct or vessel until it is unequivocally identified as the cystic structure entering the gallbladder.
9. Inflammation bailout - If dense scarring or distorted anatomy obscures Calot's triangle, convert to a subtotal "fundus-first" approach rather than blind cutting.
10. Aberrant artery caution - Preserve any large or tortuous artery (e.g., a Moynihan's hump) that might be mistaken for the cystic artery.

-----
Examples:

Example 1
Claim: "The fat and fibrous tissue overlying Calot's triangle has been fully excised, exposing only two tubular structures."
```
Category: Calot's triangle cleared
Criterion Alignment Rating: 1.0
Reasoning: The claim precisely describes complete clearance of Calot's triangle, matching this criterion perfectly.
```

Example 2
Claim: "The cystic plate is not visible due to dense adhesions, making the gallbladder-liver plane indistinct."
```
Category: Cystic plate exposed
Criterion Alignment Rating: 0.2
Reasoning: The claim refers to difficulty visualizing the cystic plate because of adhesions, which relates to this criterion but indicates failure, hence a low score.
```

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""

alignment_emotion = """You will be given a single claim that relates to why an emotion label was assigned to a piece of text. You will also be given a series of categories that an expert emotion psychologist would use to perform this type of emotion classification.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert emotion categories:
1. Valence: Decide if the overall tone is pleasant or unpleasant; positive tones suggest joy or admiration, negative tones suggest sadness or anger.
2. Arousal: Gauge how energized the wording is—calm phrasing implies low arousal emotions, intense phrasing implies high arousal emotions.
3. Emotion Words & Emojis: Look for direct emotion terms or emoticons that explicitly name the feeling.
4. Expressive Punctuation: Multiple exclamation marks, ALL‑CAPS, or stretched spellings signal higher emotional intensity.
5. Humor/Laughter Markers: Tokens like “haha,” “lol,” or laughing emojis reliably indicate amusement.
6. Confusion Phrases: Statements such as “I don’t get it” clearly mark confusion.
7. Curiosity Questions: Genuine information‑seeking phrases (“I wonder…”, “why is…?”) point to curiosity.
8. Surprise Exclamations: Reactions of astonishment (“No way!”, “I can’t believe it!”) denote surprise.
9. Threat/Worry Language: References to danger or fear (“I’m scared,” “terrifying”) signal fear or nervousness.
10. Loss or Let‑Down Words: Mentions of loss or disappointment cue sadness, disappointment, or grief.
11. Other‑Blame Statements: Assigning fault to someone else for a bad outcome suggests anger or disapproval.
12. Self‑Blame & Apologies: Admitting fault and saying “I’m sorry” marks remorse.
13. Aversion Terms: Words like “gross,” “nasty,” or “disgusting” point to disgust.
14. Praise & Compliments: Positive evaluations of someone’s actions show admiration or approval.
15. Gratitude Expressions: Phrases such as “thanks” or “much appreciated” indicate gratitude.
16. Affection & Care Words: Loving or nurturing language (“love this,” “sending hugs”) signals love or caring.
17. Self‑Credit Statements: Boasting about one’s own success (“I nailed it”) signals pride.
18. Relief Indicators: Release phrases like “phew,” “finally over,” or “what a relief” mark relief after stress ends.
-----

Here are some examples:
[Example 1]
Claim: The exclamations (“seriously wtf… sickest soulread ever”) show astonished praise for an impressive play.
Category: Surprise Exclamations
Category Alignment Rating: 0.7
Reasoning: The phrase “seriously wtf… sickest soulread ever” indicates a strong emotional reaction of astonishment. The use of “seriously” as an intensifier and the informal hyperbolic phrasing align closely with surprise. The reason the rating is 0.7 and not higher is that while there is a strong element of astonishment, the claim also suggests admiration or praise, which is more related to Praise & Compliments. Though surprise is the more dominant signal, the claim does not fully relate to surprise.

[Example 2]
Claim: The text discusses economic inflation trends over the past decade.
Category: Emotion Words & Emojis
Category Alignment Rating: 0.0
Reasoning: This claim describes factual, analytical content about economic trends and contains no reference to emotional language, emojis, or affective tone. Since the Emotion Words & Emoji category looks for direct markers of emotion (like “happy,” “sad,” “😭”), and none are present or implied in the claim, the alignment is essentially nonexistent. 

[Example 3]
Claim: The speaker is expressing happiness at a positive outcome.
Category: Valence
Category Alignment Rating: 1.0
Reasoning: The claim directly identifies the emotional tone as happiness and attributes it to a positive outcome, which maps precisely onto the Valence category. Valence is concerned with whether the emotional tone is pleasant or unpleasant, and happiness due to a good result is a prototypical example of high positive valence. There are no additional cues in the claim suggesting other categories (like specific emotion words or punctuation), so Valence is both the most relevant and strongly aligned.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""