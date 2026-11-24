"""Aspect vocabulary shared with goodtip_aspects.ipynb (copied verbatim)."""
# Moment-level aspect vocabulary for good-execution / tip sentences, drafted from a sample of training comments.
# Each comment entry (one expert, one pause, one polarity) is split into sentences; each sentence gets one
# multiple-choice question (which aspect is it about?); the softmax over the option letters is kept. Polarity is not
# asked: it comes from the list the sentence is in. "overall" and "other" are not aspects of a specific movement and
# are left out of the aspect-level analyses.

ACTIVITY = {"Basketball": "basketball", "Cooking": "cooking", "Dance": "dance", "Music": "music (playing an instrument)",
            "Rock Climbing": "bouldering (rock climbing)", "Soccer": "soccer"}

MOMENT_ASPECTS = {
    "Basketball": {
        "footwork": "footwork and steps (which foot, stride, take-off foot)",
        "legs": "knee bend, legs and landing",
        "lift": "jump, lift and elevation",
        "balance": "balance and body momentum",
        "shooting hand": "shooting hand, wrist snap and release",
        "guide hand": "guide (non-shooting) hand",
        "elbow": "shooting elbow alignment",
        "ball position": "ball placement and protection (height, position relative to the shoulders)",
        "follow-through": "follow-through and extension toward the rim",
        "eyes": "eyes, vision and focus on the target",
        "shoulders": "shoulders and body squared to the target",
        "ball flight": "ball flight and result (arc, angle, power, make or miss)",
        "dribble": "dribbling, catching and handling the ball",
        "weak hand": "use of the weaker (off) hand",
    },
    "Cooking": {
        "knife": "knife skills and cutting (grip, cuts, size of pieces)",
        "tools": "choice and use of tools and cookware",
        "heat": "heat, temperature and cooking time",
        "ingredients": "ingredients, quantities and seasoning",
        "steps": "order of steps and method (following the recipe)",
        "handling": "stirring, flipping, pouring and handling food",
        "workspace": "organization and cleanliness of the workspace",
        "safety": "safety and hygiene",
        "result": "quality of the food (doneness, texture, appearance)",
    },
    "Dance": {
        "rhythm": "rhythm and timing with the music",
        "feet": "feet, toes and steps",
        "arms": "arms and upper body",
        "legs": "legs and knees",
        "balance": "weight transfer, balance and grounding",
        "turns": "turns and spins",
        "dynamics": "energy, sharpness and dynamics",
        "clarity": "clarity and precision of the movement",
        "choreography": "memory and knowledge of the choreography",
        "style": "style, expression and personal touches",
        "space": "use of space and changes of direction",
        "ease": "ease, relaxation and flow of movement",
    },
    "Music": {
        "notes": "correct notes, pitch and intonation",
        "tempo": "tempo and rhythm",
        "tone": "tone and sound quality",
        "bowing": "bowing, strumming or picking (bow placement, weight, speed, contact point)",
        "fingering": "fingering hand (finger placement, pressure, shifting)",
        "coordination": "coordination between the hands",
        "posture": "posture, hand position, tension and relaxation",
        "musicality": "phrasing, dynamics, articulation and expression",
    },
    "Rock Climbing": {
        "footwork": "footwork and foot placement",
        "hands": "hand holds and grip",
        "hips": "hip and body positioning",
        "tension": "body tension (keeping the feet on the wall)",
        "dynamic moves": "dynamic moves and momentum",
        "route reading": "route reading and decision making",
    },
    "Soccer": {
        "touch": "first touch and ball control",
        "contact": "contact point on the ball and striking technique",
        "plant foot": "non-kicking (plant) foot and leg",
        "knee": "knee bend and leg position",
        "posture": "body posture, center of gravity and balance",
        "arms": "use of the arms",
        "movement": "movement with the ball, footwork and agility",
        "eyes": "eyes and focus on the ball",
        "weak foot": "use of the weaker foot",
    },
}

GENERIC_ASPECTS = {
    "pace": "speed, pace and efficiency",
    "fluency": "flow and continuity (without stopping or hesitating)",
    "confidence": "confidence and comfort",
    "overall": "the performance in general (a general compliment or criticism with no specific aspect)",
    "other": "none of these (describes what is happening, or is not about the performance)",
}
NON_SPECIFIC = {"overall", "other"}

def aspects_for(scenario):
    """Ordered (name, description): scenario-specific aspects, then generic ones (ending with overall, other)."""
    return [*MOMENT_ASPECTS[scenario].items(), *GENERIC_ASPECTS.items()]

