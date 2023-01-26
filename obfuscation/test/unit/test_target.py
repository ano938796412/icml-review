from main.utils.bbox import not_intersect


def test_intersect():
    """check bboxes intersect"""
    self = [2, 2, 4, 4]
    others = [[3, 3, 5, 5], [3, 1, 5, 3]]
    for other in others:
        assert not not_intersect(self, other)
        assert not not_intersect(other, self)


def test_not_intersect():
    """check bboxes don't intersect"""
    self = [2, 2, 4, 4]
    others = [[5, 3, 7, 5], [5, 1, 7, 3]]
    for other in others:
        assert not_intersect(self, other)
        assert not_intersect(other, self)
