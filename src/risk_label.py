"""Map a YOLO snake prediction to a risk label shown to the user."""

VENOMOUS = {
    "Timber Rattlesnake",
    "Eastern Diamondback Rattlesnake",
    "Pygmy Rattlesnake",
    "Eastern Copperhead",
    "Cottonmouth",
    "Eastern Coral Snake",
}

def get_risk_label(species, confidence, threshold=0.5):
    if not isinstance(species, str) or not species.strip():
        raise TypeError("species must be a non-empty string")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a number")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if confidence < threshold:
        return "Uncertain"
    return "Venomous" if species.strip().title() in VENOMOUS else "Non-venomous"
