import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recognition.textmatch import name_match, norm, number_level


def test_name_exact_and_fuzzy():
    assert name_match('Palkia', norm('BASIC Palkia HP 120'))
    assert name_match('Palkia', 'iasicpalllka120')
    assert not name_match('Charizard', norm('BASIC Palkia HP 120'))
    assert not name_match('Mew', norm('Mewtwo'))


def test_number_exact():
    assert number_level('002', 198, 'SVI EN 002/198 2023 Pokemon') == 3
    assert number_level('58', 102, 'Nintendo Creatures GAMEFREAK 58/102') == 3
    assert number_level('232', 91, 'PAF 232/091') == 3


def test_number_slash_misread():
    assert number_level('73', 82, 'Wizards 73182') == 2
    assert number_level('169', 142, 'SCR EN 1691142') == 2
    assert number_level('148', 142, 'SCR EN 14817142') == 2


def test_number_only():
    assert number_level('200', 165, 'MEW EN 200/365') == 1
    assert number_level('051', None, 'SVP EN 051') == 1


def test_number_rejects_other_cards():
    text = '1995, 96, 98 Nintendo, Creatures, GAMEFREAK 1999-2000 Wizards 21/82'
    assert number_level('21', 82, text) == 3
    assert number_level('1', 82, text) == 0
    assert number_level('8', 102, text) == 0
    assert number_level('99', 82, text) == 0
    assert number_level('21', 82, '') == 0
    assert number_level(None, 82, text) == 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
