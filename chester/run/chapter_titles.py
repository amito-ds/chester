import random

META_LEARN_EMOJIS = ["๐ค", "๐", "๐ฌ", "๐", "๐", "๐", "๐"]
FEATURE_STATS_EMOJIS = ["๐", "๐", "๐", "๐", "๐ก", "๐", "๐ง"]
FEATURE_ENGINEERING_EMOJIS = ["๐ง", "๐ฌ", "๐", "๐งฐ", "๐ก", "๐", "๐"]
MODEL_PRE_ANALYSIS_EMOJIS = ["๐", "๐", "๐ฌ", "๐งฐ", "๐ก", "๐", "๐ง"]
MODEL_RUN_EMOJIS = ["๐ค", "๐ฅ", "๐ง ", "๐", "๐ป", "๐ก", "๐ฌ"]
POST_MODEL_ANALYSIS_EMOJIS = ["๐", "๐", "๐ฌ", "๐", "๐ก", "๐", "๐"]
MODEL_WEAKNESSES_EMOJIS = ["๐ฌ", "๐", "๐ฅ", "โ", "๐", "๐", "๐ก"]
CLEAN_TEXT_EMOJIS = ["๐งผ", "๐งน", "๐ง", "๐งน", "๐", "๐ก", "๐"]


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
        emoji = '๐ต'
    message = f'{stars}\n***{prefix} {chapter_name} {emoji} ***\n{stars}'
    return message
