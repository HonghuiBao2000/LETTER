

sft_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request." \
             "\n\n### Instruction:\n{instruction}\n\n### Response:{response}"







all_prompt = {}

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. Can you help me decide?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Here are the user's historical interactions: {inters}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Based on the items that the user has interacted with: {inters}, can you determine what item would be recommended to him next?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in order: {inters}. What else do you think the user need?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Here is the item interaction history of the user: {inters}, what to recommend to the user next?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "Which item would the user be likely to interact with next after interacting with items {inters}?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "After interacting with items {inters}, what is the next item that could be recommended for the user?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "Given the user's historical interactive items arranged in chronological order: {inters}, can you recommend a suitable item for the user?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Considering the user has interacted with items {inters}. What is the next recommendation for the user?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "What is the top recommended item for the user who has previously interacted with items {inters} in order?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——12
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user most desires based on the given interaction records."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

# prompt = {}
# prompt["instruction"] = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user is most likely to interact with based on the given interaction record. Note that his most recently interacted item is {}."
# prompt["response"] = "{item}"
# prompt["task"] = "sequential"
# prompt["id"] = "1-13"
#
# seqrec_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. The historical interactions are provided as follows: {inters}."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "You can access the user's historical item interaction records: {inters}. Now your task is to recommend the next potential item to him, considering his past interactions."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "You have observed that the user has interacted with the following items: {inters}, please recommend a next item that you think would be suitable for the user."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. Using this history as a reference, please select the next item to recommend to the user."
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

all_prompt["seqrec"] = seqrec_prompt



# ========================================================
# Task 2 -- Item2Index -- 19 Prompt
# ========================================================
# Remove periods when inputting

item2index_prompt = []

# ========================================================
# Title2Index

#####——0
prompt = {}
prompt["instruction"] = "Which item has the title: \"{title}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "Which item is assigned the title: \"{title}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "An item is called \"{title}\", could you please let me know which item it is?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Which item is called \"{title}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "One of the items is named \"{title}\", can you tell me which item this is?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "What is the item that goes by the title \"{title}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

# prompt = {}
# prompt["instruction"] = "Which item is referred to as \"{title}\"?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

# ========================================================
# Description2Index

#####——6
prompt = {}
prompt["instruction"] = "An item can be described as follows: \"{description}\". Which item is it describing?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Can you tell me what item is described as \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Can you provide the item that corresponds to the following description: \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)


# prompt = {}
# prompt["instruction"] = "What is the item described as follows: \"{description}\"?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "Which item has the following characteristics: \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Which item is characterized by the following description: \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "I am curious to know which item can be described as follows: \"{description}\". Can you tell me?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

# ========================================================
# Title and Description to Index

#####——12
prompt = {}
prompt["instruction"] = "An item is called \"{title}\" and described as \"{description}\", can you tell me which item it is?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Could you please identify what item is called \"{title}\" and described as \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "Which item is called \"{title}\" and has the characteristics described below: \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "Please show me which item is named \"{title}\" and its corresponding description is: \"{description}\"."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)


# prompt = {}
# prompt["instruction"] = "Here is an item called \"{title}\" and described as \"{description}\". Which item is it?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "Determine which item this is by its title and description. The title is: \"{title}\", and the description is: \"{description}\"."
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——17
prompt = {}
prompt["instruction"] = "Based on the title: \"{title}\", and the description: \"{description}\", answer which item is this?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——18
prompt = {}
prompt["instruction"] = "Can you identify the item from the provided title: \"{title}\", and description: \"{description}\"?"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

all_prompt["item2index"] = item2index_prompt


# ========================================================
# Task 3 -- Index2Item --17 Prompt
# ========================================================
# Remove periods when inputting

index2item_prompt = []

# ========================================================
# Index2Title

#####——0
prompt = {}
prompt["instruction"] = "What is the title of item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "What title is assigned to item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} is called?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Can you provide the title of item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "What item {item} is referred to as?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Would you mind informing me about the title of item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

# ========================================================
# Index2Description

#####——6
prompt = {}
prompt["instruction"] = "Please provide a description of item {item}."
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Briefly describe item {item}."
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Can you share with me the description corresponding to item {item}?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "What is the description of item {item}?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "How to describe the characteristics of item {item}?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} looks like?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)


# ========================================================
# Index to Title and Description

#####——12
prompt = {}
prompt["instruction"] = "What is the title and description of item {item}?"
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Can you provide the corresponding title and description for item {item}?"
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "Please tell me what item {item} is called, along with a brief description of it."
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "Would you mind informing me about the title of the item {item} and how to describe its characteristics?"
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "I need to know the title and description of item {item}. Could you help me with that?"
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

all_prompt["index2item"] = index2item_prompt





# ========================================================
# Task 4 -- FusionSequentialRec -- Prompt
# ========================================================


fusionseqrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "The user has sequentially interacted with items {inters}. Can you recommend the next item for him? Tell me the title of the item？"
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "Based on the user's historical interactions: {inters}, try to predict the title of the item that the user may need next."
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Utilizing the user's past ordered interactions, which include items {inters}, please recommend the next item you think is suitable for the user and provide its title."
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)


#####——3
prompt = {}
prompt["instruction"] = "After interacting with items {inters}, what is the most probable item for the user to interact with next? Kindly provide the item's title."
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)





#####——4
prompt = {}
prompt["instruction"] = "Please review the user's historical interactions: {inters}, and describe what kind of item he still needs."
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Here is the item interaction history of the user: {inters}, please tell me what features he expects from his next item."
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "By analyzing the user's historical interactions with items {inters}, can you infer what the user's next interactive item will look like?"
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Access the user's historical item interaction records: {inters}. Your objective is to describe the next potential item for him, taking into account his past interactions."
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)






#####——8
prompt = {}
prompt["instruction"] = "Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "I possess a user's past interaction history, denoted by the title sequence of interactive items: {inter_titles}, and I am interested in knowing the user's next most desired item. Can you help me?"
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Considering the title sequence of user history interaction items: {inter_titles}. What is the next recommendation for the user?"
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "You have obtained the ordered title list of user historical interaction items, as follows: {inter_titles}. Based on this historical context, kindly choose the subsequent item for user recommendation."
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)


all_prompt["fusionseqrec"] = fusionseqrec_prompt







# ========================================================
# Task 5 -- ItemSearch -- Prompt
# ========================================================


itemsearch_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "Here is the historical interactions of a user: {inters}. And his personalized preferences are as follows: \"{explicit_preference}\". Your task is to recommend an item that is consistent with the user's preference."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "The user has interacted with a list of items, which are as follows: {inters}. Based on these interacted items, the user current intent is as follows \"{user_related_intention}\", and your task is to generate an item that matches the user's current intent."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "As a recommender system, you are assisting a user who has recently interacted with the following items: {inters}. The user expresses a desire to obtain another item with the following characteristics: \"{item_related_intention}\". Please recommend an item that meets these criteria."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Using the user's current query: \"{query}\" and his historical interactions: {inters}, you can estimate the user's preferences \"{explicit_preference}\". Please respond to the user's query by selecting an item that best matches his preference and query."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "The user needs a new item and searches for: \"{query}\". In addition, he has previously interacted with: {inters}. You can obtain his preference by analyzing his historical interactions: \"{explicit_preference}\". Can you recommend an item that best matches the search query and preferences?"
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Based on the user's historical interactions with the following items: {inters}. You can infer his preference by observing the historical interactions: \"{explicit_preference}\". Now the user wants a new item and searches for: \"{query}\". Please select a suitable item that matches his preference and search intent."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)





#####——6
prompt = {}
prompt["instruction"] = "Suppose you are a search engine, now a user searches that: \"{query}\", can you select an item to respond to the user's query?"
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "As a search engine, your task is to answer the user's query by generating a related item. The user's query is provided as \"{query}\". Please provide your generated item as your answer."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "As a recommender system, your task is to recommend an item that is related to the user's request, which is specified as follows: \"{query}\". Please provide your recommendation."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "You meet a user's query: \"{query}\". Please respond to this user by selecting an appropriate item."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)


#####——10
prompt = {}
prompt["instruction"] = "Your task is to recommend the best item that matches the user's query. Here is the search query of the user: \"{query}\", tell me the item you recommend."
prompt["response"] = "{item}"
itemsearch_prompt.append(prompt)

all_prompt["itemsearch"] = itemsearch_prompt





# ========================================================
# Task 6 -- PreferenceObtain -- Prompt
# ========================================================

preferenceobtain_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "The user has interacted with items {inters} in chronological order. Please estimate his preferences."
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "Based on the items that the user has interacted with: {inters}, can you infer what preferences he has?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Can you provide a summary of the user's preferences based on his historical interactions: {inters}?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "After interacting with items {inters} in order, what preferences do you think the user has?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Here is the item interaction history of the user: {inters}, could you please infer the user's preferences."
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "Based on the user's historical interaction records: {inters}, what are your speculations about his preferences?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Given the user's historical interactive items arranged in chronological order: {inters}, what can be inferred about the preferences of the user?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "Can you speculate on the user's preferences based on his historical item interaction records: {inters}?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "What is the preferences of a user who has previously interacted with items {inters} sequentially?"
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Using the user's historical interactions as input data, summarize the user's preferences. The historical interactions are provided as follows: {inters}."
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "Utilizing the ordered list of the user's historical interaction items as a reference, please make an informed estimation of the user's preferences. The historical interactions are as follows: {inters}."
prompt["response"] = "{explicit_preference}"
preferenceobtain_prompt.append(prompt)

all_prompt["preferenceobtain"] = preferenceobtain_prompt
