import random

CLEANING_EMOJIS = ["🧹", "🧼", "🧽", "🧴"]
PREPROCESSING_EMOJIS = ["📝", "🔍", "🔬", "🔧"]
TEXT_ANALYSIS_EMOJIS = ["🔍", "📊", "📚", "📖"]
MODEL_PRE_ANALYSIS_EMOJIS = ["📊", "📈", "🔬", "🧰"]
MODEL_RUN_EMOJIS = ["🤖", "🔥", "🧠", "🚀"]
POST_MODEL_ANALYSIS_EMOJIS = ["📊", "📉", "🔬", "📚"]
CREATE_EMBEDDING_EMOJIS = ["📚", "🔬", "🧠", "🔍"]


def chapter_message(chapter_name: str, prefix=" Running chapter: "):
    stars = '*' * (len(chapter_name + prefix) + 12)
    if chapter_name == 'cleaning':
        emoji = random.choice(CLEANING_EMOJIS)
    elif chapter_name == 'preprocessing':
        emoji = random.choice(PREPROCESSING_EMOJIS)
    elif chapter_name == 'text analyze':
        emoji = random.choice(TEXT_ANALYSIS_EMOJIS)
    elif chapter_name == 'model pre analysis':
        emoji = random.choice(MODEL_PRE_ANALYSIS_EMOJIS)
    elif chapter_name == 'model training':
        emoji = random.choice(MODEL_RUN_EMOJIS)
    elif chapter_name == 'post model analysis':
        emoji = random.choice(POST_MODEL_ANALYSIS_EMOJIS)
    elif chapter_name == 'create embedding':
        emoji = random.choice(CREATE_EMBEDDING_EMOJIS)
    else:
        emoji = '🍵'
    message = f'{stars}\n***{prefix} {chapter_name} {emoji} ***\n{stars}'
    return message



