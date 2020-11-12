
from dnips.bidict import BiDict, Ordering

def test_bidict() -> None:
    bidict = BiDict(enumerate('abcdefg', 1))

    assert list(bidict.keys()) == list(bidict.rev.values())


    assert 'a' in bidict.rev
    del bidict[1]
    assert  'a' not in bidict.rev

    bidict[1] = "A"
    assert bidict[1] == "A"
    assert bidict.rev["A"] == 1

def test_ordering() -> None:
    ordering = Ordering('abcdefg')

    assert ordering[0] == 'a'


    assert ordering.indices['a'] == 0

    assert 'abcdefg' == ''.join(i for i in ordering) 

    assert ordering == Ordering('abcdefg')
    assert ordering != Ordering('abcdeg')
