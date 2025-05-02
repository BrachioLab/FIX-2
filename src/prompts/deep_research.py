
deep_research_template = """
You are an expert in XXX. You have a deep understanding of this subject. 
Your task is to behave like an XXX and identify which criteria are important to consider for the following task:

Task description:
Input:
Output:

Here are some examples:
[Example 1]
[Example 2]
[Example 3]

Study these examples and fully understand the task. Now, research the field of XXX in order to determine a list of criteria that an expert XXX would utilize if they were performing the above task.

Your output should be a list of expert criteria, each 1 sentence long, and citations from reputable academic sources to support each criteria. Feel free to have as many expert criteria as you deem necessary. The criteria should be clear, succinct and non-overlapping with each other. [Include any domain-specific information about the expert criteria]
"""

deep_research_politeness = """
You are an expert in politeness understanding. You have a deep understanding of this subject. 
Your task is to behave like an expert linguist and identify which criteria are important to consider for the following task:

Task description: Given a short utterance from a conversation between Wikipedia editors, the task is to identify the level of politeness of the utterance. 
Input: A text utterance in English, Spanish, Japanese, or Chinese.
Output: The politeness score on a scale of 1-5, where 1=extremely rude and 5=extremely polite

Here are some examples:
[Example 1]
Input: Another way to interpret this fable is with the now cliche pride cometh before the fall. Here is another interpretation of the story; taking pride in a gift is the same thing as not having it.	
Output: 2.666

[Example 2]
Input: I saw the template citing this issue and since there was no section here discussing it I've decided to start one. I'm a Canadian and most of our television programs are also aired in the US so my knowledge of what's on TV outside of North America is limited. So I'm not sure of how much help I can be, but I do have some ideas on how to improve this section and I'm open to feedback.	
Output: 4.0

[Example 3]
Input: I agree and want to recognize your many contributions to the history of the ROC. And this is all the more reason to split into manageable articles. If there's so much content, there needs to be more articles, not clumsily keeping in one unreadable mess.	
Output: 1.333

Study these examples and fully understand the task. Now, research the field of politeness understanding in order to determine a list of criteria that an expert linguist would utilize if they were performing the above task.

Your output should be a list of expert criteria, each 1 sentence long, and citations to support each criteria. Feel free to have as many expert criteria as you deem necessary. The criteria should be clear, succinct and non-overlapping with each other. Include critera relating to lexical, syntactic, pragmatic, socio-cultural dimensions, and/or anything else you think is relevant to the task.
"""

deep_research_cholec = """
You are an expert in laparoscopic cholecystectomy. You have a deep understanding of this subject. 
Your task is to behave like an expert surgeon and identify which criteria are important to consider for the following task:

Task description:
Input: An image of the surgery site.
Output: A mask that denotes where it safe and unsafe to operate.

Please see the attached image as an example.

Study these examples and fully understand the task. Now, research the field of laparoscopic cholecystectomy in order to determine a list of criteria that an expert surgeon would utilize if they were performing the above task.

Your output should be a list of expert criteria, each 1 sentence long, and citations to support each criteria. Feel free to have as many expert criteria as you deem necessary. The criteria should be clear, succinct and non-overlapping with each other.
"""

deep_research_massmaps = """
You are an expert in cosmology. You have a deep understanding of this subject. 
Your task is to behave like an expert cosmologist and identify which criteria are important to consider for the following task:

Task description:
Input: A weak lensing map, which is the spatial distribution of matter density in the universe.
Output: A prediction for Omega_m (which captures the average energy density of all matter in the universe (relative to the total energy density which includes radiation and dark energy)) and a prediction for sigma_8 (which describes the fluctuation of matter distribution)

Here are some examples:
[Example 1] (in the first image) Omega=0.1845703125, sigma=0.9883788824081421
[Example 2] (in the second image) Omega=0.10371093451976776, sigma=1.190527319908142
[Example 3] (in the third image) Omega=0.29082030057907104, sigma=0.4727539122104645

Study these examples and fully understand the task. Now, research the field of cosmology in order to determine a list of criteria that an expert cosmologist would utilize if they were performing the above task.

Your output should be a list of expert criteria, each 1 sentence long, and citations to support each criteria. Feel free to have as many expert criteria as you deem necessary. The criteria should be clear, succinct and non-overlapping with each other.
"""