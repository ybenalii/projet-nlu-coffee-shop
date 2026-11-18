# tests/test_data.py  —  lancer avec :  make test
"""Les tests utilisent un petit jeu de données fabriqué à la volée : ils ne
dépendent pas de ce que tu as écrit dans donnees/."""
import pytest

from src.config import settings
from src.data import (read_split, get_split, load_ontology, assert_disjoint, score,
                      build_system_prompt, examples_for_prompt)

ONTO = """intentions:
  light_on: allumer
  light_off: éteindre
slots:
  room: la pièce
"""


@pytest.fixture
def donnees(tmp_path, monkeypatch):
    (tmp_path / "ontologie.yaml").write_text(ONTO, encoding="utf-8")
    (tmp_path / "train.csv").write_text(
        "phrase;reponse\nallume le salon;light_on | room=salon\n"
        "éteins la cuisine;light_off | room=cuisine\nallume tout;light_on\n", encoding="utf-8")
    (tmp_path / "validation.csv").write_text(
        "phrase;reponse\nallume la chambre;light_on | room=chambre\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text(
        "phrase;reponse\néteins le bureau;light_off | room=bureau\n", encoding="utf-8")
    monkeypatch.setattr(settings, "donnees_dir", str(tmp_path))
    return tmp_path


def test_lecture(donnees):
    assert read_split("train")[0] == ("allume le salon", "light_on | room=salon")
    assert len(get_split("train")) == 3
    assert list(load_ontology()["intentions"]) == ["light_on", "light_off"]
    assert_disjoint()


def test_fuite_detectee(donnees):
    (donnees / "test.csv").write_text("phrase;reponse\nAllume le salon;light_on | room=salon\n",
                                      encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_disjoint()


def test_prompt_depuis_ontologie(donnees):
    texte = build_system_prompt(load_ontology())
    assert "- light_on : allumer" in texte and "- room : la pièce" in texte
    assert examples_for_prompt("train", one_per_intent=True) == [
        ("allume le salon", "light_on | room=salon"), ("éteins la cuisine", "light_off | room=cuisine")]


def test_score():
    assert score("light_on | room=salon", "light_on | room=salon") == (True, 1.0, True)
    assert score("light_on | room=le salon", "light_on | room=salon")[2] is True
    io, _f1, ex = score("light_on | piece=salon", "light_on | room=salon")
    assert io is True and ex is False


def test_prompt_sans_slot(donnees):
    (donnees / "ontologie.yaml").write_text(
        "domaine: le tri des avis\nintentions:\n  positif: avis content\n  negatif:\nslots: {}\n",
        encoding="utf-8")
    texte = build_system_prompt(load_ontology())
    assert "le tri des avis" in texte and "- negatif\n" in texte and "slot" not in texte


def test_noms_importes():
    from importer import nom_propre
    assert nom_propre("Card linking") == "card_linking"
    assert nom_propre("Négatif") == "negatif"
    assert nom_propre(3) == "classe_3"
