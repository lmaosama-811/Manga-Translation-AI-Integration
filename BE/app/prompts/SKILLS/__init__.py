from . import action, comedy, historical, mysterious 

SKILLS = {
    "action": action.SKILL,
    "comedy": comedy.SKILL,
    "historical": historical.SKILL,
    "mysterious": mysterious.SKILL
}

def load_skill(genre: str) -> str:
    return SKILLS.get(genre, "")