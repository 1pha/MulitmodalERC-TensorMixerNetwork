from itertools import chain

from erc.preprocess import get_folds


def _test_fold_split(num_session, num_folds, gt_sess: list):
    """ Test if `get_folds` function with followings
    1. Different sessions, different folds
    2. Range overlap
    """
    assert sum(gt_sess) == num_session,\
        f"GT is not correctly set. #gt_sess: {sum(gt_sess)} != num_session: {num_session}"\
        f"\nTest not correctly set."
    
    fold_dict = get_folds(num_session=num_session, num_folds=num_folds)
    assert len(fold_dict) == num_folds,\
        f"fold_dict should contain {num_folds} folds. Given: {len(fold_dict)}"
    num_sess = [r.stop - r.start for r in fold_dict.values()]
    assert num_sess == gt_sess, f"Split sessions should "

    sess = set(chain.from_iterable([list(r) for r in fold_dict.values()]))
    assert len(sess) == num_session, f""


def test_fold_split():
    # KEMDy19
    num_session, num_folds, gt_sess = 20, 5, [4, 4, 4, 4, 4]
    _test_fold_split(num_session=num_session, num_folds=num_folds, gt_sess=gt_sess)
    num_session, num_folds, gt_sess = 20, 7, [3, 3, 3, 3, 3, 3, 2]
    _test_fold_split(num_session=num_session, num_folds=num_folds, gt_sess=gt_sess)

    # KEMDy20_v1_1
    num_session, num_folds, gt_sess = 40, 5, [8, 8, 8, 8, 8]
    _test_fold_split(num_session=num_session, num_folds=num_folds, gt_sess=gt_sess)
    num_session, num_folds, gt_sess = 40, 7, [6, 6, 6, 6, 6, 5, 5]
    _test_fold_split(num_session=num_session, num_folds=num_folds, gt_sess=gt_sess)