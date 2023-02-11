import random

META_LEARN_EMOJIS = ["🤖", "📊", "🔬", "🚀", "🔍", "📉", "📈"]
FEATURE_STATS_EMOJIS = ["📈", "📉", "📊", "🔍", "💡", "📚", "🔧"]
FEATURE_ENGINEERING_EMOJIS = ["🔧", "🔬", "🔍", "🧰", "💡", "📚", "📊"]
MODEL_PRE_ANALYSIS_EMOJIS = ["📊", "📈", "🔬", "🧰", "💡", "📚", "🔧"]
MODEL_RUN_EMOJIS = ["🤖", "🔥", "🧠", "🚀", "💻", "💡", "🔬"]
POST_MODEL_ANALYSIS_EMOJIS = ["📊", "📉", "🔬", "📚", "💡", "📈", "📉"]
MODEL_WEAKNESSES_EMOJIS = ["🔬", "📉", "💥", "❌", "🔍", "📈", "💡"]
CLEAN_TEXT_EMOJIS = ["🧼", "🧹", "💧", "🧹", "📚", "💡", "🔍"]


def chapter_title(chapter_name: str, prefix=" Chapter: "):
    stars = '*' * (len(chapter_name + prefix) + 12)
    if chapter_name == 'meta learn':
        emoji = random.choice(META_LEARN_EMOJIS)
    elif chapter_name == 'feature statistics':
        emoji = random.choice(FEATURE_STATS_EMOJIS)
    elif chapter_name == 'feature engineering':
        emoji = random.choice(FEATURE_ENGINEERING_EMOJIS)
    elif chapter_name == 'model pre analysis':
        emoji = random.choice(MODEL_PRE_ANALYSIS_EMOJIS)
    elif chapter_name == 'model training':
        emoji = random.choice(MODEL_RUN_EMOJIS)
    elif chapter_name == 'post model analysis':
        emoji = random.choice(POST_MODEL_ANALYSIS_EMOJIS)
    elif chapter_name == 'model weaknesses':
        emoji = random.choice(MODEL_WEAKNESSES_EMOJIS)
    elif chapter_name == 'model weaknesses':
        emoji = random.choice(MODEL_WEAKNESSES_EMOJIS)
    elif chapter_name == 'text cleaning':
        emoji = random.choice(CLEAN_TEXT_EMOJIS)
    else:
        emoji = '🍵'
    message = f'{stars}\n***{prefix} {chapter_name} {emoji} ***\n{stars}'
    return message
