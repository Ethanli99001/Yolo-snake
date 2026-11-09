import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pytest
from risk_label import get_risk_label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Normal cases
def test_venomous_species():
    assert get_risk_label("Cottonmouth", 0.92) == "Venomous"

def test_non_venomous_species():
    assert get_risk_label("Northern Watersnake", 0.88) == "Non-venomous"

def test_look_alike_is_not_venomous():
    # Mimic of the coral snake
    assert get_risk_label("Scarlet Kingsnake", 0.95) == "Non-venomous"


# Low confidence: never tell the user a snake is safe when unsure
def test_low_confidence_venomous_is_uncertain():
    assert get_risk_label("Cottonmouth", 0.30) == "Uncertain"


# Decimal boundary values around the threshold
@pytest.mark.parametrize("conf, expected", [
    (0.4999, "Uncertain"),
    (0.5, "Venomous"),
    (0.5001, "Venomous"),
    (0.0, "Uncertain"),
    (1.0, "Venomous"),
])
def test_threshold_boundaries(conf, expected):
    assert get_risk_label("Timber Rattlesnake", conf) == expected


# Out-of-range confidence
@pytest.mark.parametrize("conf", [-0.2, 1.5])
def test_out_of_range_confidence(conf):
    with pytest.raises(ValueError):
        get_risk_label("Cottonmouth", conf)


# Invalid input types
@pytest.mark.parametrize("conf", ["0.9", None, True])
def test_invalid_confidence_type(conf):
    with pytest.raises(TypeError):
        get_risk_label("Cottonmouth", conf)

@pytest.mark.parametrize("species", ["", "   ", None, 123])
def test_invalid_species(species):
    with pytest.raises(TypeError):
        get_risk_label(species, 0.9)


# Case/whitespace in species name
def test_species_name_normalized():
    assert get_risk_label("  cottonmouth ", 0.9) == "Venomous"
