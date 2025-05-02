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


alignment_poltieness = """You will be given a single claim that relates to why a politeness rating was given to an utterance. You will also be given a series of categories that an expert linguist would use to perform this type of politeness classification.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Lexical categories:
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


alignment_cholec = """
You will be given:
- An endoscopic image of the gallbladder region during a laparoscopic cholecystectomy (the "Input").
- A textual Claim describing a proposed "safe" or "unsafe" zone in that image (the "Claim").

Your task is as follows:
1. **Determine which one expert safety criterion** (from the list below) the Claim most relates to.
2. **Rate how strongly** the Claim aligns with that criterion on a scale from 0.0 to 1.0 (in increments of 0.1), where 0.0 means "not at all" and 1.0 means "perfectly."

Return your answer exactly in this format:
```
Criterion: <name of selected criterion>
Alignment Rating: <0.0-1.0>
Reasoning: <A brief explanation of why this criterion was chosen and how you judged the alignment score>
```

-----
**Expert Safety Criteria**
1. **Calot's triangle cleared** - Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
2. **Cystic plate exposed** - The lower third of the gallbladder must be dissected off the liver to reveal the shiny cystic plate and ensure correct plane.
3. **Only two structures visible** - Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
4. **Above the R4U line** - Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to segment IV to avoid the common bile duct.
5. **Infundibulum start point** - Dissection should begin at the gallbladder infundibulum-cystic duct junction to stay in safe tissue planes.
6. **Subserosal plane stay** - When separating the gallbladder from the liver, stay in the avascular subserosal cleavage plane under the serosal fat layer.
7. **Cystic lymph node guide** - Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.
8. **No division without ID** - Never divide any duct or vessel until it is unequivocally identified as the cystic structure entering the gallbladder.
9. **Inflammation bailout** - If dense scarring or distorted anatomy obscures Calot's triangle, convert to a subtotal "fundus-first" approach rather than blind cutting.
10. **Aberrant artery caution** - Preserve any large or tortuous artery (e.g., a Moynihan's hump) that might be mistaken for the cystic artery.

-----
**Examples**

**Example 1**
Input: [image of a well-cleared triangle]
Claim: "The fat and fibrous tissue overlying Calot's triangle has been fully excised, exposing only two tubular structures."
```
Criterion: Calot's triangle cleared
Alignment Rating: 1.0
Reasoning: The claim exactly describes the complete clearance of Calot's triangle so that only the cystic duct and artery remain visible, which matches this criterion perfectly.
```

**Example 2**
Input: [image showing thick adhesions around gallbladder]
Claim: "There is dense fibrosis obscuring the cystic plate, making it hard to see the liver-gallbladder plane."
```
Criterion: Cystic plate exposed
Alignment Rating: 0.2
Reasoning: The claim refers to difficulty visualizing the cystic plate due to fibrosis; it relates to this criterion but indicates a failure rather than achievement, hence a low alignment score.
```

Now evaluate the following:

Input: (see attached image),
Claim: [[CLAIM]]
"""