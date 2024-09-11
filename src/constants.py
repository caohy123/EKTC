# SYSTEM_PROMPT = "You are a multimodal empathetic conversational AI chatbot that can empathize with users and use stickers to assist in empathy when appropriate."
SYSTEM_PROMPT = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible. You are an empathetic conversational AI chatbot that can empathize with users and use emotion knowledge base  tool to assist in empathy when appropriate. You only need to provide the next round of response of Listener."
# SYSTEM_PROMPT = "This is an empathetic dialogue task: The first worker (Speaker) is given an emotion label and writes his own description of a situation when he has felt that way. Then, Speaker tells his story in a conversation with a second worker (Listener). The emotion label and situation of Speaker are invisible to Listener. Listener should recognize and acknowledge others’ feelings in a conversation as much as possible.\
#   Now you play the role of Listener, please give the corresponding response according to the existing context. You only need to provide the next round of response of Listener."
USER_PREFIX = "human"
ASSISTANT_PREFIX = "gpt"
FUNCTION_PREFIX = "function_call"
OBSERVATION_PREFIX = "observation"
THOUGHT_PREFIX = "thought"

THOUGHT_PROMPT = "I need to generate a sticker."

generate_sticker_tool = dict(
    name = "GenerateSticker",
    description = "generate a sticker with prompt", 
    parameters = dict(
        type = "object",
        properties = dict(
            prompt = dict(
                description = "the description of sticker that you want to generate."
            )
        ),
        required = ["prompt"]
    )
)

generate_relation_tool = dict(
    name = "EmotionKnowledgeBase",
    description = "generate a relation with prompt", 
    parameters = dict(
        type = "object",
        properties = dict(
            prompt = dict(
                description = "the emotional relation in the dialogue context that you want to generate."
            )
        ),
        required = ["prompt"]
    )
)
